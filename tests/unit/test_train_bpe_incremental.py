"""Guard merge semantics when replacing full rescans with incremental updates."""

import random
from collections import Counter

import pytest

from lm_from_scratch.tokenization.train_bpe import count_chunk, train_bpe


@pytest.mark.parametrize(
    "text, expected",
    [
        ("", []),
        ("a", []),
        ("aaaaa", [(b"a", b"a"), (b"aa", b"aa"), (b"aaaa", b"a")]),
        ("abababab", [(b"a", b"b"), (b"ab", b"ab"), (b"abab", b"abab")]),
        ("ab<|endoftext|>ac", [(b"a", b"c"), (b"a", b"b")]),
        ("ab<|endoftext|>ab<|endoftext|>az", [(b"a", b"b"), (b"a", b"z")]),
    ],
)
def test_merge_overlaps_weights_and_ties(tmp_path, text, expected):
    path = tmp_path / "corpus.txt"
    path.write_text(text, encoding="utf-8")
    vocab, merges = train_bpe(path, 300, ["<|endoftext|>"], num_workers=1)
    assert merges == expected
    assert [vocab[i] for i in range(257, len(vocab))] == [a + b for a, b in expected]


def _rescan_reference(path, target):
    """Deliberately slow oracle: derive every pair count anew each iteration."""
    counts = count_chunk((path, 0, path.stat().st_size, ("<|endoftext|>",)))
    vocab = {i: bytes([i]) for i in range(256)}
    vocab[256] = b"<|endoftext|>"
    merges = []
    while len(vocab) < target:
        pairs = Counter()
        for pieces, frequency in counts.items():
            for pair in zip(pieces, pieces[1:]):
                pairs[pair] += frequency
        if not pairs:
            break
        winner = max(pairs, key=lambda pair: (pairs[pair], pair))
        merges.append(winner)
        vocab[len(vocab)] = b"".join(winner)
        updated = Counter()
        for pieces, frequency in counts.items():
            result = []
            i = 0
            while i < len(pieces):
                if pieces[i : i + 2] == winner:
                    result.append(b"".join(winner))
                    i += 2
                else:
                    result.append(pieces[i])
                    i += 1
            updated[tuple(result)] += frequency
        counts = updated
    return vocab, merges


@pytest.mark.parametrize("seed", range(20))
def test_incremental_matches_full_rescan(tmp_path, seed):
    # Repeated documents, self-overlap, Unicode bytes, and equal counts exercise
    # stale priorities, changing pair membership, and frequency-weighted deltas.
    rng = random.Random(seed)
    documents = ["".join(rng.choices("aaaabbbcdefé界🙂 ", k=rng.randrange(1, 45))) for _ in range(25)]
    documents += documents[:10] * 3
    path = tmp_path / "corpus.txt"
    path.write_text("<|endoftext|>".join(documents), encoding="utf-8")
    assert train_bpe(path, 400, ["<|endoftext|>"], num_workers=1) == _rescan_reference(path, 400)
