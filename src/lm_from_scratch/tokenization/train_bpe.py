from typing import List
import regex as re
import pickle
from pathlib import Path
from collections import Counter, defaultdict
import heapq
from multiprocessing import get_context
import os 
import argparse
import json
from threading import Event, Thread
from time import perf_counter

import psutil
from tqdm import tqdm

PRETOKEN_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class _PairCandidate:
    """Make heapq choose the greatest (frequency, byte-pair), as max did."""

    __slots__ = ("frequency", "pair")

    def __init__(self, frequency, pair):
        self.frequency = frequency
        self.pair = pair

    def __lt__(self, other):
        if self.frequency != other.frequency:
            return self.frequency > other.frequency
        return self.pair > other.pair

def find_chunk_boundaries(file, desired_num_chunks, split_special_token):
    "Return byte offsets that align with document boundaries"
    if desired_num_chunks < 1:
        raise ValueError("desired num_chunks must be positive.")

    if not isinstance(split_special_token, bytes) or not split_special_token:
        raise ValueError("split_special_token must be nonempty bytes.")
    
    # find the file's size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    
    boundaries = {0, file_size}
    for index in range(1, desired_num_chunks):
        position = index * file_size // desired_num_chunks
        file.seek(position)
        
        while True:
            line = file.readline()
            
            # Reached the end of the file
            if not line:
                boundaries.add(file_size)
                break
            
            marker_position = line.find(split_special_token)
            
            # Split token index found
            if marker_position != -1:
                boundaries.add(position + marker_position)
                break
                
            position += len(line)
    
    return sorted(boundaries)

def count_chunk(task):
    """
    WORKER FUNCTION
    
    Read one byte range and return its pre-token frequency table.
    This function does not learn BPE merges.
    """
    input_path, start, end, special_tokens = task
    
    # Each worker opens its own file handle.
    with open(input_path, "rb") as file:
        file.seek(start)
        chunk_bytes = file.read(end - start)
    
    text = chunk_bytes.decode('utf-8')
    
    # Remove special tokens while preserving document boundaries
    if special_tokens:
        SPECIAL_TOKEN_PAT = "|".join(
            re.escape(special_token) for special_token in sorted(special_tokens, key=len, reverse=True)
        )
        segments = re.split(SPECIAL_TOKEN_PAT, text)
    else:
        segments = [text]

    counts = Counter()
    
    for segment in segments:
        for match in re.finditer(PRETOKEN_PAT, segment):
            encoded = match.group().encode('utf-8')
            
            # A key is a tuple of single-byte bytes objects
            pieces = tuple(bytes([byte]) for byte in encoded)
            counts[pieces] += 1
    
    return counts 
    
def count_pretokens_parallel(
    input_path, special_tokens, num_workers, split_special_token=None,
    show_progress=False,
):
    """
    PARENT FUNCTION.
    
    Divide the file into tasks, send them to the workers, and combine counts.
    """
    if num_workers < 1:
        raise ValueError("num_workers must be positive.")

    if num_workers == 1 or not special_tokens:
        file_size = Path(input_path).stat().st_size
        
        with tqdm(total=1, desc="Pretokenizing", unit="chunk", disable=not show_progress) as progress:
            counts = count_chunk((input_path, 0, file_size, special_tokens))
            progress.update(1)
        return counts
    
    # The regex uses strings; binary file boundary detection needs ONE byte marker.
    if split_special_token is None:
        split_special_token = special_tokens[0].encode("utf-8")
    if not isinstance(split_special_token, bytes) or not split_special_token:
        raise ValueError("split_special_token must be nonempty bytes.")
    if split_special_token.decode("utf-8") not in special_tokens:
        raise ValueError("The chunk delimiter must be a registered special token.")
    if b"\n" in split_special_token:
        raise ValueError("The line-based chunk finder needs a delimiter without newlines.")

    # More chunks than workers lets workers take another task when done.
    num_chunks = num_workers * 4
    
    with open(input_path, "rb") as file:
        boundaries = find_chunk_boundaries(
            file,
            desired_num_chunks=num_chunks,
            split_special_token=split_special_token
        )
    
    tasks = [
        (input_path, start, end, tuple(special_tokens))
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]
    
    total_counts = Counter()
    
    context = get_context("spawn")
    
    with (
        context.Pool(processes=num_workers) as pool,
        tqdm(total=len(tasks), desc="Pretokenizing", unit="chunk", disable=not show_progress) as progress,
    ):
        for chunk_counts in pool.imap_unordered(
            count_chunk,
            tasks,
            chunksize=1
        ):
            total_counts.update(chunk_counts)
            progress.update(1)
    
    return total_counts
    
    
def train_bpe(
    input_path: str, 
    vocab_size: int, 
    special_tokens: List[str],
    num_workers: int = 4,
    show_progress: bool = False,
):
    # 1. Vocabulary dictionary initialization [0, 255]
    vocabs = { idx: bytes([idx]) for idx in range(256) }
    
    special_tokens = list(dict.fromkeys(special_tokens))
    for special_token in special_tokens:
        if not special_token:
            raise ValueError("Special tokens must not be empty.")

        token_bytes = special_token.encode("utf-8")
        if token_bytes not in vocabs.values():
            vocabs[len(vocabs)] = token_bytes

    if vocab_size < len(vocabs):
        raise ValueError(
            f"vocab_size must be at least {len(vocabs)} "
            "to hold the initial vocabulary."
        )
    
    # 2. Build ONE global frequency table using worker processes
    pretoken_frequency = count_pretokens_parallel(
        input_path=input_path,
        special_tokens=special_tokens,
        num_workers=num_workers,
        show_progress=show_progress,
    )
    
    # 3. Count pairs once and index the pretokens containing each pair.
    # Stable IDs let us replace sequences without rebuilding the global table.
    merges = []
    if len(vocabs) == vocab_size:
        return vocabs, merges

    pretokens = list(pretoken_frequency)
    frequencies = list(pretoken_frequency.values())
    del pretoken_frequency
    pair_frequency = Counter()
    pair_to_pretokens = defaultdict(set)
    for pretoken_id, pieces in enumerate(pretokens):
        for pair, occurrences in Counter(zip(pieces, pieces[1:])).items():
            pair_frequency[pair] += occurrences * frequencies[pretoken_id]
            pair_to_pretokens[pair].add(pretoken_id)

    candidates = [_PairCandidate(frequency, pair) for pair, frequency in pair_frequency.items()]
    heapq.heapify(candidates)

    # 4. Learn merges in the parent process, updating only affected pretokens.
    for _ in tqdm(
        range(vocab_size - len(vocabs)),
        desc="Learning BPE merges",
        unit="merge",
        disable=not show_progress,
    ):
        # Count changes leave old heap entries behind. Discard stale priorities.
        while candidates:
            candidate = heapq.heappop(candidates)
            if pair_frequency.get(candidate.pair, 0) == candidate.frequency:
                best_pair = candidate.pair
                break
        else:
            if show_progress:
                tqdm.write("No adjacent pairs remain; stopping before the vocabulary target.")
            break
        
        merged_bytes = best_pair[0] + best_pair[1]
        
        vocabs[len(vocabs)] = merged_bytes
        merges.append(best_pair)
        
        affected_ids = pair_to_pretokens.pop(best_pair)
        changed_pairs = set()
        for pretoken_id in affected_ids:
            pieces = pretokens[pretoken_id]
            frequency = frequencies[pretoken_id]
            before = Counter(zip(pieces, pieces[1:]))
            updated_pieces = []
            position = 0 
            
            while position < len(pieces):
                if (position + 1 < len(pieces) and 
                    (pieces[position], pieces[position+1]) == best_pair
                ):
                    updated_pieces.append(merged_bytes)
                    position += 2
                else:
                    updated_pieces.append(pieces[position])
                    position += 1

            updated_pieces = tuple(updated_pieces)
            pretokens[pretoken_id] = updated_pieces
            after = Counter(zip(updated_pieces, updated_pieces[1:]))

            # Count the actual before/after sequences to handle self-overlap
            # (e.g. a a a) and multiple occurrences with the original weighting.
            for pair in before.keys() | after.keys():
                delta = (after[pair] - before[pair]) * frequency
                if delta:
                    pair_frequency[pair] += delta
                    if pair_frequency[pair] == 0:
                        del pair_frequency[pair]
                    changed_pairs.add(pair)

            for pair in before.keys() - after.keys():
                members = pair_to_pretokens.get(pair)
                if members is not None:  # The winning pair was already popped.
                    members.remove(pretoken_id)
                    if not members:
                        del pair_to_pretokens[pair]
            for pair in after.keys() - before.keys():
                pair_to_pretokens[pair].add(pretoken_id)

        # Publish only the final counts after all affected pretokens are updated.
        for pair in changed_pairs:
            frequency = pair_frequency.get(pair, 0)
            if frequency:
                heapq.heappush(candidates, _PairCandidate(frequency, pair))

        # Bound memory from stale priorities on long runs.
        if len(candidates) > max(1024, 4 * len(pair_frequency)):
            candidates = [_PairCandidate(frequency, pair) for pair, frequency in pair_frequency.items()]
            heapq.heapify(candidates)
    
    return vocabs, merges
                    
def main():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Train the TinyStories BPE tokenizer.")
    parser.add_argument(
        "--input", type=Path,
        default=repo_root / "data" / "TinyStoriesV2-GPT4-train.txt",
    )
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-progress", action="store_true", help="Hide training progress bars.")
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).resolve().parent / "artifacts" / "tinystories_train",
    )
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"Input file does not exist: {args.input}")
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.vocab_size < 257:
        parser.error("--vocab-size must be at least 257")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    special_tokens = ["<|endoftext|>"]

    # Sample the parent and its workers together, rather than measuring only
    # the parent. This is sampled RSS, not an exact peak or unique memory use.
    process = psutil.Process()
    stop_sampling = Event()
    peak_rss = [0]
    sampling_errors = []

    def sample_memory():
        while True:
            rss = 0
            try:
                for member in [process, *process.children(recursive=True)]:
                    try:
                        rss += member.memory_info().rss
                    except psutil.NoSuchProcess:
                        pass  # A worker can exit between enumeration and sampling.
            except (psutil.Error, OSError) as error:
                sampling_errors.append(str(error))
                return
            peak_rss[0] = max(peak_rss[0], rss)
            if stop_sampling.wait(0.1):
                break

    print(
        f"Training on {args.input.resolve()}\n"
        f"Vocabulary target: {args.vocab_size}; workers: {args.workers}",
        flush=True,
    )
    monitor = Thread(target=sample_memory, daemon=True)
    monitor.start()
    started = perf_counter()
    try:
        vocab, merges = train_bpe(
            input_path=args.input,
            vocab_size=args.vocab_size,
            special_tokens=special_tokens,
            num_workers=args.workers,
            show_progress=not args.no_progress,
        )
    finally:
        elapsed = perf_counter() - started
        stop_sampling.set()
        monitor.join()

    vocab_path = args.output_dir / "tinystories_vocab.pkl"
    merges_path = args.output_dir / "tinystories_merges.pkl"
    with vocab_path.open("wb") as file:
        pickle.dump(vocab, file)
    with merges_path.open("wb") as file:
        pickle.dump(merges, file)

    longest_id, longest_bytes = max(vocab.items(), key=lambda item: len(item[1]))
    metrics = {
        "input_path": str(args.input.resolve()),
        "input_bytes": args.input.stat().st_size,
        "requested_vocab_size": args.vocab_size,
        "actual_vocab_size": len(vocab),
        "merge_count": len(merges),
        "workers": args.workers,
        "special_tokens": special_tokens,
        "training_seconds": elapsed,
        "peak_sampled_process_tree_rss_bytes": None if sampling_errors else peak_rss[0],
        "memory_sampling_errors": sampling_errors,
        "memory_measurement": (
            "Maximum sampled sum of parent and descendant RSS, every 0.1s; "
            "may miss brief peaks and count shared pages more than once."
        ),
        "longest_token_id": longest_id,
        "longest_token_length_bytes": len(longest_bytes),
        "longest_token_bytes_repr": repr(longest_bytes),
        "longest_token_hex": longest_bytes.hex(),
        "longest_token_text": longest_bytes.decode("utf-8", errors="replace"),
    }
    metrics_path = args.output_dir / "training_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Training time: {elapsed:.2f} seconds")
    if sampling_errors:
        print(f"Memory measurement unavailable: {sampling_errors[0]}")
    else:
        print(f"Peak sampled process-tree RSS: {peak_rss[0] / 1024**3:.3f} GiB")
    print(f"Vocabulary entries: {len(vocab)}; merges: {len(merges)}")
    print(f"Longest token ({len(longest_bytes)} bytes): {longest_bytes!r}")
    print(f"Saved {vocab_path}\nSaved {merges_path}\nSaved {metrics_path}")


if __name__ == "__main__":
    main()
