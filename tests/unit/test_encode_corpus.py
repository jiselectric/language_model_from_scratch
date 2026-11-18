import pickle
import struct
import subprocess
import sys


def test_corpus_encoding_command_preserves_utf8_bytes(tmp_path):
    vocab_path = tmp_path / "vocab.pkl"
    merges_path = tmp_path / "merges.pkl"
    vocab_path.write_bytes(pickle.dumps({index: bytes([index]) for index in range(256)}))
    merges_path.write_bytes(pickle.dumps([]))
    corpus = tmp_path / "corpus.txt"
    corpus.write_bytes("A\né".encode())
    output = tmp_path / "tokens.bin"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "lm_from_scratch.tokenization.encode_corpus",
            "--vocab",
            str(vocab_path),
            "--merges",
            str(merges_path),
            "--input",
            str(corpus),
            "--output",
            str(output),
        ],
        check=True,
    )
    assert output.read_bytes() == struct.pack("<4H", 65, 10, 195, 169)
