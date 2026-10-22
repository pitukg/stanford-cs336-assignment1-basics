from pathlib import Path
import pickle

from cs336_basics.tokenizer import train_bpe


if __name__ == "__main__":
    vocab, merges = train_bpe(
        Path(__file__).parent.parent / "data/TinyStoriesV2-GPT4-train.txt",
        vocab_size=10_000,
        special_tokens=["<|endoftext|>"],
        num_workers=4
    )

    with open(
        Path(__file__).parent.parent /
        "data/encoding/TinyStoriesV2-GPT4-train-vocab.pk",
        "wb"
    ) as f:
        pickle.dump(vocab, f)
        print("Dumped vocab")


    with open(
        Path(__file__).parent.parent /
        "data/encoding/TinyStoriesV2-GPT4-train-merges.pk",
        "wb"
    ) as f:
        pickle.dump(merges, f)
        print("Dumped merges")
