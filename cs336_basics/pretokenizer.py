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

def get_pretokens_counter(chunk: str, special_tokens: list[str] | None
                         ) -> Counter[tuple[bytes, ...]]:
    # Split on special tokens
    if special_tokens is not None and len(special_tokens) > 0:
        sub_chunks = re.split(r"|".join(map(re.escape, special_tokens)), chunk)
    else:
        sub_chunks = [chunk]
    counter = Counter()
    for sub_chunk in sub_chunks:
        for match in re.finditer(PAT, sub_chunk):
            counter[
                tuple(map(
                    lambda b: bytes([b]),
                    match.group(0).encode(ENCODING)
                ))
            ] += 1
    return counter


def get_pretokens_list(chunk: str, special_tokens: list[str] | None
                      ) -> list[tuple[bytes, ...]]:
    # Split on special tokens
    prev = 0
    if special_tokens is not None and len(special_tokens) > 0:
        special_matches = re.finditer(
            r"|".join(map(re.escape, special_tokens)),
            chunk,
            re.POSIX
        )
    else:
        special_matches = []
    pretokens = []
    for next_special_match in special_matches:
        for match in re.finditer(PAT, chunk[prev:next_special_match.start(0)]):
            pretokens.append(
                tuple(map(
                    lambda b: bytes([b]),
                    match.group(0).encode(ENCODING)
                ))
            )
        # Append special
        pretokens.append((next_special_match.group(0).encode(ENCODING),))
        prev = next_special_match.end(0)
    for match in re.finditer(PAT, chunk[prev:]):
        pretokens.append(
            tuple(map(
                lambda b: bytes([b]),
                match.group(0).encode(ENCODING)
            ))
        )
    return pretokens


if __name__ == "__main__":
    # get_pretokens_counter("Here is some 24.36 asd.<endoftext>Here is another!<endoftext>",
    print(get_pretokens_list("Here is some 24.36 asd.<endoftext>Here is another!<endoftext>",
                  ["<endoftext>"]))
