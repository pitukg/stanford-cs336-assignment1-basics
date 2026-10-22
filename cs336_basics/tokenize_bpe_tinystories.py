import codecs
import itertools
import numpy as np
import pathlib
from tqdm import tqdm
from typing import Iterable

from cs336_basics import pretokenizer
from cs336_basics.tokenizer import Tokenizer


if __name__ == "__main__":
    tokenizer = Tokenizer.from_files(
        vocab_filepath=pathlib.Path(__file__).parent.parent / \
            "data/encoding/TinyStoriesV2-GPT4-train-vocab.pk",
        merges_filepath=pathlib.Path(__file__).parent.parent / \
            "data/encoding/TinyStoriesV2-GPT4-train-merges.pk",
        special_tokens=["<|endoftext|>"]
    )
    def yield_chunks(filepath_in, chunk_size=4_096) -> Iterable[str]:
        size = filepath_in.stat().st_size
        decoder_cls = codecs.getincrementaldecoder(pretokenizer.ENCODING)
        decoder = decoder_cls(errors="strict")
        with open(
            filepath_in,
            "rb"
        ) as f_in:
            with tqdm(
                    total=size, unit="B", unit_scale=True,
                    unit_divisor=1_024) as bar:
                while (byte_chunk := f_in.read(chunk_size)) != b'':
                    read_len = len(byte_chunk)
                    str_chunk = decoder.decode(byte_chunk)
                    del byte_chunk
                    yield str_chunk
                    bar.update(read_len)
                str_chunk = decoder.decode(b'', final=True)
                yield str_chunk

    chunk_iterator = yield_chunks(
        pathlib.Path(__file__).parent.parent /
                "data/TinyStoriesV2-GPT4-debug.txt")

    with open(
        pathlib.Path(__file__).parent.parent /
            "data/encoding/TinyStoriesV2-GPT4-debug-tokenized.np",
        "wb"
    ) as f_out:
        for token_ids in itertools.batched(
                tokenizer.encode_iterable(chunk_iterator), n=2_048):
            token_ids_np = np.array(token_ids)
            assert np.all(0 <= token_ids_np) and np.all(token_ids_np < 65_535)
            token_ids_np = token_ids_np.astype("<u2")
            token_ids_np.tofile(f_out)
            del token_ids
            del token_ids_np
