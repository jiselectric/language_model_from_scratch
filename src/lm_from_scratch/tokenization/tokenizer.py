import pickle
from pathlib import Path
import regex as re
from itertools import islice
import numpy as np

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab.copy()
        self.vocabInversion = {byte_sequence: id for id, byte_sequence in self.vocab.items()}
        self.merges = merges
        self.merge_ranks = { pair: idx for idx, pair in enumerate(self.merges)}
        
        if not special_tokens:
            special_tokens = []
        self.special_tokens = special_tokens
        self.special_tokens_set = set(self.special_tokens)
        
        for special_token in special_tokens:
            encoded = special_token.encode('utf-8')
            if encoded not in self.vocabInversion: 
                new_id = max(self.vocab, default=-1) + 1
                self.vocab[new_id] = encoded
                self.vocabInversion[encoded] = new_id

    @classmethod
    def from_files(cls, vocab_filepath=None, merges_filepath=None, special_tokens=None):
        with open(vocab_filepath, "rb") as file:
            vocab = pickle.load(file)
        
        with open(merges_filepath, "rb") as file:
            merges = pickle.load(file)
        
        tokenizer = cls(vocab, merges, special_tokens)
        
        return tokenizer
        
    def encode(self, text):
        PRETOKEN_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        if self.special_tokens:
            SPECIAL_TOKEN_PAT = '|'.join(
                re.escape(special_token) for special_token in sorted(self.special_tokens, key=len, reverse=True)
            )
            segments = re.split(f"({SPECIAL_TOKEN_PAT})", text)
        else:
            segments = [text]
        
        ids = []
        for segment in segments:
            if not segment:
                continue
                
            if segment in self.special_tokens_set:
                ids.append(self.vocabInversion[segment.encode('utf-8')])
                continue
            
            for match in re.finditer(PRETOKEN_PAT, segment):
                pieces = [bytes([byte]) for byte in match.group().encode('utf-8')] 
                
                while len(pieces) > 1:
                    best_index = None
                    best_rank = float('inf')
                    
                    for index in range(len(pieces) - 1):
                        pair = (pieces[index], pieces[index+1])
                        rank = self.merge_ranks.get(pair, float('inf'))
                        
                        if rank < best_rank:
                            best_index = index
                            best_rank = rank
                    
                    if best_index is None:
                        break
                    pieces[best_index:best_index+2] = [pieces[best_index] + pieces[best_index+1]]
                
                ids.extend(self.vocabInversion[piece] for piece in pieces)
        
        return ids
                
    def encode_iterable(self, iterable):
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

            special_matches = (
                special_pattern.finditer(pending)
                if special_pattern
                else ()
            )

            # Find safe boundaries before and at complete special tokens.
            for special in special_matches:
                for match in pretoken_pattern.finditer(
                    pending, start, special.start()
                ):
                    if (
                        match.end() <= safe_limit
                        and not pending[match.end() - 1].isspace()
                    ):
                        cut = match.end()

                if (
                    special.end() <= safe_limit
                    and not pending[special.end() - 1].isspace()
                ):
                    cut = special.end()

                start = special.end()

            # Inspect ordinary text after the last special token.
            for match in pretoken_pattern.finditer(pending, start):
                if (
                    match.end() <= safe_limit
                    and not pending[match.end() - 1].isspace()
                ):
                    cut = match.end()

            if cut:
                ready = pending[:cut]
                pending = pending[cut:]

                yield from self.encode(ready)

        # End of input: nothing else can extend the remaining text.
        if pending:
            yield from self.encode(pending)            
            
    def decode(self, ids):
        sequences = []
        for id in ids:
            sequences.append(self.vocab[id])
        
        return b''.join(sequences).decode('utf-8', errors='replace')

if __name__ == "__main__":
    artifact_dir = Path(__file__).resolve().parent / "artifacts" / "tinystories_train"
    tokenizer = Tokenizer.from_files(
        artifact_dir / "tinystories_vocab.pkl",
        artifact_dir / "tinystories_merges.pkl",
        ["<|endoftext|>"]
    )

    data_dir = Path(__file__).resolve().parent.parent / "data"
    input_path = data_dir / "TinyStoriesV2-GPT4-train.txt"
    output_path = data_dir / "tinystories_train_tokens.bin"

    with (
        input_path.open("r", encoding="utf-8", newline="") as reader,
        output_path.open("wb") as writer,
    ):
        token_ids = tokenizer.encode_iterable(reader)

        while True:
            batch = np.fromiter(
                islice(token_ids, 100_000),
                dtype="<u2",
            )

            if batch.size == 0:
                break

            batch.tofile(writer)
