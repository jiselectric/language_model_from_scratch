import os
import pickle
from collections.abc import Iterable, Iterator

import regex as re


class Tokenizer:
    """Byte-level BPE with atomic special tokens and chunked text encoding."""

    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ) -> None:
        self.vocab = vocab.copy()
        self.vocab_inverse = {byte_sequence: token_id for token_id, byte_sequence in self.vocab.items()}
        self.merges = list(merges)
        self.merge_ranks = {pair: idx for idx, pair in enumerate(self.merges)}

        if not special_tokens:
            special_tokens = []
        self.special_tokens = list(special_tokens)
        self.special_tokens_set = set(self.special_tokens)

        for special_token in special_tokens:
            encoded = special_token.encode("utf-8")
            if encoded not in self.vocab_inverse:
                new_id = max(self.vocab, default=-1) + 1
                self.vocab[new_id] = encoded
                self.vocab_inverse[encoded] = new_id

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | os.PathLike[str],
        merges_filepath: str | os.PathLike[str],
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        """Load vocabulary and merges from trusted pickle files."""
        with open(vocab_filepath, "rb") as file:
            vocab = pickle.load(file)

        with open(merges_filepath, "rb") as file:
            merges = pickle.load(file)

        tokenizer = cls(vocab, merges, special_tokens)

        return tokenizer

    def encode(self, text: str) -> list[int]:
        pretoken_pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        if self.special_tokens:
            special_token_pattern = "|".join(
                re.escape(special_token) for special_token in sorted(self.special_tokens, key=len, reverse=True)
            )
            segments = re.split(f"({special_token_pattern})", text)
        else:
            segments = [text]

        ids = []
        for segment in segments:
            if not segment:
                continue

            if segment in self.special_tokens_set:
                ids.append(self.vocab_inverse[segment.encode("utf-8")])
                continue

            for match in re.finditer(pretoken_pattern, segment):
                pieces = [bytes([byte]) for byte in match.group().encode("utf-8")]

                while len(pieces) > 1:
                    best_index = None
                    best_rank = float("inf")

                    for index in range(len(pieces) - 1):
                        pair = (pieces[index], pieces[index + 1])
                        rank = self.merge_ranks.get(pair, float("inf"))

                        if rank < best_rank:
                            best_index = index
                            best_rank = rank

                    if best_index is None:
                        break
                    pieces[best_index : best_index + 2] = [pieces[best_index] + pieces[best_index + 1]]

                ids.extend(self.vocab_inverse[piece] for piece in pieces)

        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        pretoken_pattern = re.compile(
            r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+|"
            r" ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
        )

        special_pattern = None
        if self.special_tokens:
            special_pattern = re.compile(
                "|".join(
                    re.escape(token)
                    for token in sorted(
                        self.special_tokens,
                        key=len,
                        reverse=True,
                    )
                )
            )

        # Keep enough trailing characters for an unfinished contraction
        # or a special token split across incoming chunks.
        lookahead = max(
            3,
            max(
                (len(token) - 1 for token in self.special_tokens),
                default=0,
            ),
        )

        pending = ""
        for text in iterable:
            pending += text
            safe_limit = len(pending) - lookahead

            if safe_limit <= 0:
                continue

            cut = 0
            start = 0

            special_matches = special_pattern.finditer(pending) if special_pattern else ()

            # Find safe boundaries before and at complete special tokens.
            for special in special_matches:
                for match in pretoken_pattern.finditer(pending, start, special.start()):
                    if match.end() <= safe_limit and not pending[match.end() - 1].isspace():
                        cut = match.end()

                if special.end() <= safe_limit and not pending[special.end() - 1].isspace():
                    cut = special.end()

                start = special.end()

            # Inspect ordinary text after the last special token.
            for match in pretoken_pattern.finditer(pending, start):
                if match.end() <= safe_limit and not pending[match.end() - 1].isspace():
                    cut = match.end()

            if cut:
                ready = pending[:cut]
                pending = pending[cut:]

                yield from self.encode(ready)

        # End of input: nothing else can extend the remaining text.
        if pending:
            yield from self.encode(pending)

    def decode(self, ids: Iterable[int]) -> str:
        sequences = []
        for token_id in ids:
            sequences.append(self.vocab[token_id])

        return b"".join(sequences).decode("utf-8", errors="replace")
