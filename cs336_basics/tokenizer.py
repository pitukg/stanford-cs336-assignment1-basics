from collections import Counter
import pathlib

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


def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str],
              num_workers: int = 1
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
    with open(input_path, "rb") as f:
        boundaries = pretokenizer.find_chunk_boundaries(f, num_workers, END_OF_TEXT)

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        frequencies_by_chunk: list[Counter[tuple[bytes, ...]]] = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # Run pre-tokenization on your chunk and store the counts for each pre-token
            frequencies_by_chunk.append(
                pretokenizer.get_pretokens(chunk, special_tokens))

    frequencies = sum(frequencies_by_chunk, Counter())
    token_pairs_count = count_token_pairs(frequencies, num_workers)
    frequencies = {
        b''.join(key): (freq, key) for key, freq in frequencies.items()
    }

    vocab = [bytes([i]) for i in range(256)]
    merges: list[tuple[bytes, bytes]] = []
    while len(special_tokens) + len(vocab) < vocab_size:
        frequencies, token_pairs_count, merge = do_merge(
            frequencies, token_pairs_count, num_workers)
        merges.append(merge)
        vocab.append(b''.join(merge))

    return (
        {**dict(enumerate(vocab)), **{
            len(vocab) + i: spec.encode(pretokenizer.ENCODING)
            for i, spec in enumerate(special_tokens)
        }},
        merges
    )


if __name__ == "__main__":
    single_worker = train_bpe(pathlib.Path(__file__).parent.parent / "data/TinyStoriesV2-GPT4-debug.txt", 1000, ["<|endoftext|>"])
    # multi_worker = train_bpe(pathlib.Path(__file__).parent.parent / "data/TinyStoriesV2-GPT4-debug.txt", 1000, ["<|endoftext|>"], num_workers=8)
    # assert single_worker == multi_worker
