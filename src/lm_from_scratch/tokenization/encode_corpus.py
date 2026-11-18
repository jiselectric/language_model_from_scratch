"""Stream a UTF-8 corpus into little-endian uint16 token IDs."""

import argparse
import struct
from itertools import islice
from pathlib import Path

from lm_from_scratch.tokenization.tokenizer import Tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument("--merges", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--special-token", action="append", default=[])
    args = parser.parse_args()
    tokenizer = Tokenizer.from_files(args.vocab, args.merges, args.special_token)
    if args.input.resolve() == args.output.resolve():
        parser.error("Input and output must be different files.")
    if any(token_id < 0 or token_id > 65535 for token_id in tokenizer.vocab):
        parser.error("Vocabulary IDs must fit in uint16 (0 through 65535).")
    with args.input.open("r", encoding="utf-8", newline="") as reader, args.output.open("wb") as writer:
        token_ids = tokenizer.encode_iterable(reader)
        while batch := list(islice(token_ids, 100_000)):
            # Explicit endianness makes files portable across host architectures.
            writer.write(struct.pack(f"<{len(batch)}H", *batch))


if __name__ == "__main__":
    main()
