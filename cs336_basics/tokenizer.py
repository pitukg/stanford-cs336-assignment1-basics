from concurrent.futures import ProcessPoolExecutor
from collections import Counter
from functools import partial
import pathlib
import pickle
from typing import Iterable, Iterator
from tqdm import tqdm

from cs336_basics import pretokenizer


def count_token_pairs(freqs: Counter[tuple[bytes, ...]], num_workers: int = 1
                     ) -> dict[tuple[bytes, bytes]]:
    assert num_workers == 1
    token_pairs_count = Counter()
    for pretoken, freq in freqs.items():
        for token_pair in zip(pretoken[:-1], pretoken[1:]):
            token_pairs_count[token_pair] += freq
    assert token_pairs_count, "Expecting token pairs"
    return token_pairs_count

def do_merge(freqs: dict[bytes, tuple[int, tuple[bytes, ...]]],
             token_pairs_count: Counter[tuple[bytes, bytes]],
             num_workers: int = 1
            ) -> tuple[
                dict[bytes, tuple[int, tuple[bytes, ...]]],
                Counter[tuple[bytes, bytes]],
                tuple[bytes, bytes]
            ]:
    assert num_workers == 1
    most_common = token_pairs_count.most_common()
    merge_candidates = []
    for pair, count in most_common:
        if count < most_common[0][1]:
            break
        merge_candidates.append(pair)
    merge = max(merge_candidates)
    aux_merge = b''.join(merge)
    merged_freqs = freqs
    def apply_merge(token: tuple[bytes, ...], merge: tuple[bytes, bytes],
                    aux_merge: bytes, freq: int,
                    joint_token_bytes: bytes | None
                   ) -> tuple[tuple[bytes, ...], Counter[tuple[bytes, bytes]]]:
        # return new merges
        if joint_token_bytes is not None and \
                aux_merge not in joint_token_bytes:
            return token, None
        for i in range(len(token)-1):
            if token[i:i+2] == merge:
                recursed_merge, recursed_new_token_pairs = apply_merge(
                    token[i+2:], merge, aux_merge, freq, None)
                ret = (
                    *token[:i],
                    b''.join(merge),
                    *recursed_merge
                )
                recursed_new_token_pairs = Counter() \
                    if recursed_new_token_pairs is None else recursed_new_token_pairs
                if i > 0:
                    recursed_new_token_pairs[(token[i-1], b''.join(merge))] += freq
                    recursed_new_token_pairs[(token[i-1], merge[0])] -= freq
                if len(recursed_merge) > 0:
                    recursed_new_token_pairs[(b''.join(merge), recursed_merge[0])] += freq
                    recursed_new_token_pairs[(merge[1], token[i+2])] -= freq
                recursed_new_token_pairs[merge] -= freq
                return ret, recursed_new_token_pairs
        return token, None
    merged_token_pairs_count = token_pairs_count
    for joint_token_bytes, (freq, token) in freqs.items():
        merged_token, new_merges = apply_merge(token, merge, aux_merge, freq,
                                               joint_token_bytes)
        if merged_token != token:
            merged_freqs[joint_token_bytes] = (freq, merged_token)
            if new_merges is not None:
                merged_token_pairs_count.update(new_merges)

    return merged_freqs, merged_token_pairs_count, merge


def pretokenize_chunk(start, end, input_path, special_tokens):
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        # Run pre-tokenization on your chunk and store the counts for each pre-token
        return pretokenizer.get_pretokens_counter(chunk, special_tokens)

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str],
              num_workers: int = 8
             ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Deliverable: Write a function that, given a path to an input text file,
    trains a (byte-level) BPE tokenizer.

    Input
    input_path: str Path to a text file with BPE tokenizer training data.
    vocab_size: int A positive integer that defines the maximum final
                    vocabulary size (including the initial byte vocabulary,
                    vocabulary items produced from merging, and any special
                    tokens).
    special_tokens: list[str] A list of strings to add to the vocabulary.
                            During training, treat them as hard boundaries
                            that prevent merges across their spans, but do
                            not include them when computing merge statistics.

    Output
    vocab: dict[int, bytes] The tokenizer vocabulary, a mapping from int
                            (token ID in the vocabulary) to bytes (token bytes).
    merges: list[tuple[bytes, bytes]] A list of BPE merges produced from
                                    training. Each list item is a tuple of
                                    bytes (<token1>, <token2>), representing
                                    that <token1> was merged with <token2>.
                                    The merges should be ordered by order
                                    of creation.
    """
    END_OF_TEXT = b"<|endoftext|>"
    assert END_OF_TEXT.decode("utf-8") in special_tokens
    with open(input_path, "rb") as f_boundaries:
        boundaries = pretokenizer.find_chunk_boundaries(f_boundaries, num_workers, END_OF_TEXT)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        print(f"Starting pretokenization with {num_workers} workers")
        frequencies_by_chunk: list[Counter[tuple[bytes, ...]]] = list(
            executor.map(
                partial(
                    pretokenize_chunk,
                    input_path=input_path,
                    special_tokens=special_tokens
                ),
                boundaries[:-1],
                boundaries[1:]
            )
        )

    frequencies = sum(frequencies_by_chunk, Counter())
    print("Counting token pairs")
    token_pairs_count = count_token_pairs(frequencies, num_workers=1)
    frequencies = {
        b''.join(key): (freq, key) for key, freq in frequencies.items()
    }

    vocab = [bytes([i]) for i in range(256)]
    merges: list[tuple[bytes, bytes]] = []
    print("Starting BPE merges...")
    for _ in tqdm(list(range(vocab_size - len(special_tokens) - len(vocab)))):
    # while len(special_tokens) + len(vocab) < vocab_size:
        frequencies, token_pairs_count, merge = do_merge(
            frequencies, token_pairs_count, num_workers=1)
        merges.append(merge)
        vocab.append(b''.join(merge))

    return (
        {**dict(enumerate(vocab)), **{
            len(vocab) + i: spec.encode(pretokenizer.ENCODING)
            for i, spec in enumerate(special_tokens)
        }},
        merges
    )


class Tokenizer:
    def __init__(self,
            vocab: dict[int, bytes],
            merges: list[tuple[bytes, bytes]],
            special_tokens: list[str] | None = None
    ):
        """
        Construct a tokenizer from a given vocabulary, list of merges,
        and (optionally) a list of special tokens.
        """
        self.vocab = vocab
        self.merges: list[tuple[tuple[bytes, bytes], bytes]] = [
            (merge, b''.join(merge))
            for merge in merges
        ]
        self.special_tokens = special_tokens
        if special_tokens:
            for special_token in special_tokens:
                if (
                    enc_special_token := special_token.encode(pretokenizer.ENCODING)
                ) not in vocab.values():
                    self.vocab[len(self.vocab)] = enc_special_token
        self.reverse_vocab: dict[bytes, int] = {
            value: key for key, value in self.vocab.items()
        }
        self.max_special_length = max(map(len, map(
                lambda s: s.encode(pretokenizer.ENCODING), self.special_tokens))) \
            if self.special_tokens is not None \
                and len(self.special_tokens) > 0 \
            else 0

    @classmethod
    def from_files(cls,
                   vocab_filepath: str,
                   merges_filepath: str,
                   special_tokens: list[str] | None = None
                  ):
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        # Pretokenize the text
        pretokens = pretokenizer.get_pretokens_list(text, self.special_tokens)
        pretokens = [(pretoken, b''.join(pretoken)) for pretoken in pretokens]
        return self._encode_pretokens(pretokens)

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        prefix, last_pretoken = "", None
        for chunk in iterable:
            pretokens = pretokenizer.get_pretokens_list(
                prefix + chunk, self.special_tokens)
            pretokens = [(pretoken, b''.join(pretoken)) for pretoken in pretokens]
            # Last pretokens might be partial, save for next turn
            # Count back max special length
            dropped, dropped_list = 0, []
            while dropped < self.max_special_length and len(pretokens) > 0:
                _, dropped_joined_pretoken = pretokens.pop()
                dropped += len(dropped_joined_pretoken)
                dropped_list.append(dropped_joined_pretoken)
            prefix = b''.join(reversed(dropped_list)).decode(
                pretokenizer.ENCODING)
            if len(pretokens) > 0:
                if len(pretokens) > 1:
                    for token_id in self._encode_pretokens(pretokens[:-1]):
                        yield token_id
                prefix = pretokens[-1][1].decode(pretokenizer.ENCODING) + prefix
            # print(f"Keeping {prefix=} for next turn...")
        pretokens = pretokenizer.get_pretokens_list(
            prefix, self.special_tokens)
        joined_pretokens = [(pretoken, b''.join(pretoken)) for pretoken in pretokens]
        for token_id in self._encode_pretokens(joined_pretokens):
            yield token_id

    def _encode_pretokens(self, pretokens: list[tuple[tuple[bytes, ...], bytes]]):
        def apply_merge(pretoken: tuple[bytes, ...], merge: tuple[bytes, bytes],
                        aux_pretoken: bytes | None, aux_merge: bytes
                        ):
            if aux_pretoken is not None and aux_merge not in aux_pretoken:
                return pretoken
            for i in range(len(pretoken)-1):
                if pretoken[i:i+2] == merge:
                    recursed_merge = apply_merge(
                        pretoken[i+2:], merge, None, aux_merge)
                    ret = (
                        *pretoken[:i],
                        aux_merge,
                        *recursed_merge
                    )
                    return ret
            return pretoken
        for merge, aux_merge in self.merges:
            for i in range(len(pretokens)):
                pretoken, aux_pretoken = pretokens[i]
                pretokens[i] = (apply_merge(pretoken, merge, aux_pretoken, aux_merge), aux_pretoken)
        return [
            self.reverse_vocab[token_bytes]
            for pretoken, _ in pretokens
            for token_bytes in pretoken
        ]

    def decode(self, ids: list[int]) -> str:
        return b''.join([self.vocab[id] for id in ids]).decode(
            pretokenizer.ENCODING, errors='replace')


if __name__ == "__main__":
    # single_worker = train_bpe(pathlib.Path(__file__).parent.parent / "data/TinyStoriesV2-GPT4-debug.txt", 1000, ["<|endoftext|>"])
    # multi_worker = train_bpe(pathlib.Path(__file__).parent.parent / "data/TinyStoriesV2-GPT4-debug.txt", 1000, ["<|endoftext|>"], num_workers=8)
    # assert single_worker == multi_worker
    tokenizer = Tokenizer.from_files(
        vocab_filepath=pathlib.Path(__file__).parent.parent / \
            "data/encoding/TinyStoriesV2-GPT4-train-vocab.pk",
        merges_filepath=pathlib.Path(__file__).parent.parent / \
            "data/encoding/TinyStoriesV2-GPT4-train-merges.pk",
        special_tokens=["<|endoftext|>", "<|endoftext|><|endoftext|>"]
    )
    text_chunks = ["Hiya, I'm doing this.<|endo","fte","xt|> and that other<|endoftext|>","<|endoftext|>"]
    token_ids = list(tokenizer.encode_iterable(text_chunks))
    print(f"{token_ids=}")
    single_token_ids = list(tokenizer.encode("".join(text_chunks)))
    assert tuple(token_ids) == tuple(single_token_ids)
    decoded = tokenizer.decode(token_ids)
    print(f"{decoded=}")
    assert decoded == "".join(text_chunks)
    pass
