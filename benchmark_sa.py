"""
benchmark_sa.py
===============
Final checkpoint benchmarking suite for all SA builders.

Measures and compares across all four builders:
  sorted   — pure-Python O(N² log N)
  sais     — pure-Python O(N log² N) prefix-doubling
  rust     — infini-gram Rust SAIS (single-threaded, O(N))
  caps_sa  — CaPS-SA parallel (multi-threaded, O(N))

Metrics collected per (builder, dataset_size):
  - Index construction time (total and SA-only)
  - Peak memory usage (MB)
  - Query latency: median and p99 for 1-gram, 2-gram, 4-gram queries

Corpus sizes (tokens):
  Small  :   500,000  (~1MB)
  Medium : 5,000,000  (~10MB)
  Large  : 50,000,000 (~100MB)
  (sorted and sais are skipped above 1M tokens — too slow)

Usage:
    # run all builders, all sizes
    python benchmark_sa.py

    # run specific builders (including naive Python builders)
    python benchmark_sa.py --builders sorted sais rust caps_sa

    # run specific sizes (in tokens)
    python benchmark_sa.py --sizes 500000 5000000

    # set thread count for parallel builders
    python benchmark_sa.py --threads 8

    # save results to CSV
    python benchmark_sa.py --output results/benchmark.csv

    # keep index files on disk after benchmarking
    python benchmark_sa.py --index-dir /scratch/indices/
"""

import os
import csv
import sys
import time
import random
import struct
import shutil
import argparse
import tempfile
import resource
import numpy as np
from typing import List, Tuple, Optional

from infini_gram import (
    build_index,
    tokenize_documents,
    InfiniGramEngine,
    _RUST_BIN,
    _CAPS_BIN,
)


# ─── Corpus sizes ─────────────────────────────────────────────────────────────

# Builders too slow to run above this token count
NAIVE_MAX_TOKENS = 1_000_000

DEFAULT_SIZES = [500_000, 5_000_000, 50_000_000]


# ─── Corpus Generation ────────────────────────────────────────────────────────

def generate_corpus(n_tokens: int, vocab_size: int = 1000) -> List[str]:
    """
    Generate a synthetic corpus of approximately n_tokens total tokens
    using a Zipf distribution to simulate realistic token frequencies.
    Documents are ~500 tokens each.
    """
    print(f"  [corpus] Generating corpus: ~{n_tokens:,} tokens, "
          f"vocab_size={vocab_size}, doc_len=500 ...")
    t0 = time.perf_counter()

    rng     = np.random.default_rng(42)
    doc_len = 500
    n_docs  = max(1, n_tokens // doc_len)

    weights = 1.0 / np.arange(1, vocab_size + 1)
    weights /= weights.sum()
    token_ids = rng.choice(
        np.arange(1, vocab_size + 1),
        size=n_docs * doc_len,
        p=weights,
    )

    str_tokens = token_ids.astype(str).reshape(n_docs, doc_len)
    docs = [' '.join(row) for row in str_tokens]

    print(f"  [corpus] Done: {len(docs):,} docs in {time.perf_counter()-t0:.2f}s")
    return docs


# ─── Memory Tracking ──────────────────────────────────────────────────────────

def peak_memory_mb() -> float:
    """Return peak RSS memory usage in MB."""
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage / 1e3 if sys.platform != 'darwin' else usage / 1e6


# ─── Query Latency ────────────────────────────────────────────────────────────

def measure_query_latency(
    engine: InfiniGramEngine,
    flat: List[int],
    n_gram: int,
    reps: int = 200,
) -> Tuple[float, float]:
    """
    Measure median and p99 query latency for random n-gram lookups.
    Returns (median_us, p99_us).
    """
    positions = [
        i for i in random.sample(range(len(flat) - n_gram),
                                  min(500, len(flat) - n_gram))
        if 0 not in flat[i:i + n_gram]
    ][:50]

    if not positions:
        return 0.0, 0.0

    times = []
    for _ in range(reps):
        pos   = random.choice(positions)
        query = flat[pos:pos + n_gram]
        t0    = time.perf_counter()
        engine.count(query)
        times.append((time.perf_counter() - t0) * 1e6)

    times.sort()
    return float(np.median(times)), times[int(len(times) * 0.99)]


# ─── Single Benchmark Run ─────────────────────────────────────────────────────

def run_benchmark(
    builder:   str,
    n_tokens:  int,
    index_dir: str,
    n_threads: int = 1,
) -> Optional[dict]:
    """
    Run a single benchmark for one (builder, corpus_size) combination.
    Returns a results dict, or None if the run was skipped.
    """
    # skip naive builders on large corpora
    if builder in ('sorted', 'sais') and n_tokens > NAIVE_MAX_TOKENS:
        print(f"  [skip] {builder} skipped for {n_tokens:,} tokens "
              f"(limit: {NAIVE_MAX_TOKENS:,})")
        return None

    print(f"\n  Builder: {builder}  |  ~{n_tokens:,} tokens  |  "
          f"threads: {n_threads}")
    print(f"  {'-'*52}")

    docs       = generate_corpus(n_tokens)
    mem_before = peak_memory_mb()

    # ── Tokenization ──────────────────────────────────────────────────────────
    print(f"  [tokenize] Tokenizing {len(docs):,} documents ...")
    t_tok_start = time.perf_counter()

    # ── Index construction ────────────────────────────────────────────────────
    print(f"  [index] Building index with builder={builder!r} ...")
    t0   = time.perf_counter()
    meta = build_index(
        docs, index_dir,
        eos_token_id=0,
        sa_builder=builder,
        n_threads=n_threads,
    )
    t_total    = time.perf_counter() - t0
    mem_after  = peak_memory_mb()
    actual_tok = meta['n_tokens']

    print(f"  [index] Tokenization : {meta['t_tokenize']:.3f}s")
    print(f"  [index] SA build     : {meta['t_sa']:.3f}s  "
          f"({actual_tok:,} tokens → "
          f"{actual_tok * 5 / 1e6:.1f} MB table.bin)")
    print(f"  [index] Total        : {t_total:.3f}s")
    print(f"  [index] Peak memory  : {mem_after - mem_before:.1f} MB")

    # ── Reconstruct flat token list for query benchmarks ──────────────────────
    print(f"  [query] Reconstructing flat token list for query benchmarks ...")
    tokenized_docs, _ = tokenize_documents(docs, eos_token_id=0)
    flat = []
    for toks in tokenized_docs:
        flat.extend(toks)
        flat.append(0)

    # ── Load engine ───────────────────────────────────────────────────────────
    print(f"  [query] Loading InfiniGramEngine from {index_dir} ...")
    engine = InfiniGramEngine(index_dir, eos_token_id=0)
    print(f"  [query] Engine ready: {engine.n_tokens:,} tokens, "
          f"{engine.n_sa:,} SA entries")

    # ── Query latency ─────────────────────────────────────────────────────────
    latencies = {}
    for n in [1, 2, 4]:
        print(f"  [query] Running {n}-gram latency benchmark (200 reps × 50 queries) ...")
        med, p99 = measure_query_latency(engine, flat, n_gram=n)
        latencies[n] = (med, p99)
        print(f"  [query] {n}-gram  →  median={med:.1f}µs  p99={p99:.1f}µs")

    engine.close()
    print(f"  [done] Builder={builder!r} finished.")

    return {
        'builder':    builder,
        'n_tokens':   actual_tok,
        'n_threads':  n_threads,
        't_total_s':  round(t_total, 3),
        't_sa_s':     round(meta['t_sa'], 3),
        'mem_mb':     round(mem_after - mem_before, 1),
        'lat_1g_med': round(latencies[1][0], 2),
        'lat_1g_p99': round(latencies[1][1], 2),
        'lat_2g_med': round(latencies[2][0], 2),
        'lat_2g_p99': round(latencies[2][1], 2),
        'lat_4g_med': round(latencies[4][0], 2),
        'lat_4g_p99': round(latencies[4][1], 2),
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Benchmark SA builders across corpus sizes'
    )
    parser.add_argument('--builders', nargs='+',
                        default=['rust', 'caps_sa'],
                        choices=['sorted', 'sais', 'rust', 'caps_sa'],
                        help='Builders to benchmark '
                             '(default: rust caps_sa — use --builders sorted sais '
                             'to include the naive Python builders)')
    parser.add_argument('--sizes', nargs='+', type=int,
                        default=DEFAULT_SIZES,
                        help='Corpus sizes in tokens '
                             '(default: 500k 5M 50M)')
    parser.add_argument('--threads', type=int, default=1,
                        help='Thread count for parallel builders '
                             '(default: 1)')
    parser.add_argument('--output', type=str, default=None,
                        help='CSV file to write results to')
    parser.add_argument('--index-dir', type=str, default=None,
                        help='Base directory for index files '
                             '(default: temp, cleaned up after each run)')
    args = parser.parse_args()

    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        # write header if file doesn't exist yet
        if not os.path.exists(args.output):
            with open(args.output, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'builder','n_tokens','n_threads','t_total_s','t_sa_s',
                    'mem_mb','lat_1g_med','lat_1g_p99','lat_2g_med','lat_2g_p99',
                    'lat_4g_med','lat_4g_p99'
                ])
                writer.writeheader()

    # check external binaries
    if 'rust' in args.builders and not os.path.isfile(_RUST_BIN):
        print(f"WARNING: rust_indexing not found at {_RUST_BIN} — skipping")
        args.builders = [b for b in args.builders if b != 'rust']
    if 'caps_sa' in args.builders and not os.path.isfile(_CAPS_BIN):
        print(f"WARNING: CaPS-SA not found at {_CAPS_BIN} — skipping")
        args.builders = [b for b in args.builders if b != 'caps_sa']

    if not args.builders:
        print("No valid builders available. Exiting.")
        sys.exit(1)

    print(f"Builders : {args.builders}")
    print(f"Sizes    : {[f'{s:,}' for s in sorted(args.sizes)]}")
    print(f"Threads  : {args.threads}")

    random.seed(42)
    all_results = []

    for n_tokens in sorted(args.sizes):
        print(f"\n{'='*60}")
        print(f"Corpus size: ~{n_tokens:,} tokens")
        print(f"{'='*60}")

        for builder in args.builders:
            use_tmp = args.index_dir is None
            if use_tmp:
                index_dir = tempfile.mkdtemp(
                    prefix=f'bench_{builder}_{n_tokens}_')
            else:
                index_dir = os.path.join(
                    args.index_dir, f'{builder}_{n_tokens}')
                os.makedirs(index_dir, exist_ok=True)

            try:
                result = run_benchmark(
                    builder   = builder,
                    n_tokens  = n_tokens,
                    index_dir = index_dir,
                    n_threads = args.threads,
                )
                if result is not None:
                    all_results.append(result)
                    # write immediately after each run
                    if args.output:
                        with open(args.output, 'a', newline='') as f:
                            writer = csv.DictWriter(f, fieldnames=result.keys())
                            writer.writerow(result)
                        print(f"  [csv] Result saved to {args.output}")
            except Exception as e:
                print(f"  ERROR: {builder} @ {n_tokens:,} tokens: {e}")
            finally:
                if use_tmp:
                    shutil.rmtree(index_dir, ignore_errors=True)

    if not all_results:
        print("\nNo results collected.")
        sys.exit(1)

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n\n{'='*95}")
    print(f"{'BENCHMARK SUMMARY':^95}")
    print(f"{'='*95}")
    print(
        f"{'Builder':<10} {'Tokens':>12} {'t_total(s)':>12} {'t_SA(s)':>10} "
        f"{'Mem(MB)':>9} {'1g_med':>8} {'1g_p99':>8} "
        f"{'2g_med':>8} {'4g_med':>8}"
    )
    print('-' * 95)
    for r in all_results:
        print(
            f"{r['builder']:<10} {r['n_tokens']:>12,} {r['t_total_s']:>12.3f} "
            f"{r['t_sa_s']:>10.3f} {r['mem_mb']:>9.1f} "
            f"{r['lat_1g_med']:>8.1f} {r['lat_1g_p99']:>8.1f} "
            f"{r['lat_2g_med']:>8.1f} {r['lat_4g_med']:>8.1f}"
        )

    # ── CSV output ────────────────────────────────────────────────────────────
    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nResults written to {args.output}")


if __name__ == '__main__':
    main()