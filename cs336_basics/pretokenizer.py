from collections import Counter
import os
import regex as re
from typing import BinaryIO


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
ENCODING = "utf-8"

def get_pretokens(chunk: str, special_tokens: list[str]
                 ) -> Counter[tuple[bytes, ...]]:
    # Split on special tokens
    sub_chunks = re.split(r"|".join(map(re.escape, special_tokens)), chunk)
    sub_counters = []
    for sub_chunk in sub_chunks:
        sub_counters.append(Counter())
        for match in re.finditer(PAT, sub_chunk):
            sub_counters[-1][
                tuple(map(
                    lambda b: bytes([b]),
                    match.group(0).encode(ENCODING)
                ))
            ] += 1
    counter = sum(sub_counters, Counter())
    return counter


if __name__ == "__main__":
    get_pretokens("Here is some 24.36 asd.<endoftext>Here is another!<endoftext>",
                  ["<endoftext>"])
