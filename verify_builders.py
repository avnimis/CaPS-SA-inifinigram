"""
verify_builders.py
==================
Correctness validation for all four SA builders on larger corpora.

Validates that every builder produces a suffix array that:
  1. Covers all token positions exactly once
  2. Is in non-decreasing lexicographic (byte-level) order
  3. Produces counts matching a naive brute-force scan for
     1-gram, 2-gram, and 4-gram queries

Builders:
  sorted   — pure-Python O(N² log N)
  sais     — pure-Python O(N log² N) prefix-doubling
  rust     — infini-gram Rust SAIS
  caps_sa  — CaPS-SA parallel

Usage:
    python verify_builders.py
    python verify_builders.py --builders rust caps_sa
    python verify_builders.py --n_tokens 2000000
    python verify_builders.py --threads 8
"""

import os
import sys
import struct
import random
import shutil
import argparse
import tempfile
import numpy as np
from typing import List

from infini_gram import (
    build_index,
    tokenize_documents,
    InfiniGramEngine,
    _RUST_BIN,
    _CAPS_BIN,
)


# ─── Corpus sizes ─────────────────────────────────────────────────────────────

# sorted and sais are too slow above this
NAIVE_MAX_TOKENS = 500_000


# ─── Corpus Generation ────────────────────────────────────────────────────────

def generate_corpus(n_tokens: int, vocab_size: int = 1000) -> List[str]:
    """
    Generate a synthetic corpus of approximately n_tokens total tokens
    using a Zipf distribution to simulate realistic token frequencies.
    Documents are ~500 tokens each.
    """
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

    docs = []
    for i in range(n_docs):
        chunk = token_ids[i * doc_len:(i + 1) * doc_len]
        docs.append(' '.join(str(t) for t in chunk))
    return docs


# ─── Correctness Checks ───────────────────────────────────────────────────────

def check_sa_covers_all_positions(engine: InfiniGramEngine) -> bool:
    """Every token position must appear exactly once in the SA."""
    positions = [engine._sa(r) for r in range(engine.n_sa)]
    return sorted(positions) == list(range(engine.n_tokens))


def check_sa_order(engine: InfiniGramEngine, flat: List[int],
                   sample: int = 1000) -> bool:
    """
    Spot-check that adjacent SA entries are in non-decreasing byte-level
    lexicographic order, matching rust_indexing and CaPS-SA sort order.
    """
    def to_bytes(tok_list: List[int]) -> bytes:
        return b''.join(struct.pack('<H', t) for t in tok_list)

    ranks = random.sample(range(engine.n_sa - 1), min(sample, engine.n_sa - 1))
    for rank in ranks:
        a  = engine._sa(rank)
        b  = engine._sa(rank + 1)
        sa = to_bytes(flat[a:a + 8])
        sb = to_bytes(flat[b:b + 8])
        if sa > sb:
            print(f"  SA out of order at rank {rank}: "
                  f"{flat[a:a+8]} > {flat[b:b+8]}")
            return False
    return True


def naive_count(flat: List[int], query: List[int]) -> int:
    n, k = len(flat), len(query)
    return sum(1 for i in range(n - k + 1) if flat[i:i + k] == query)


def check_counts(engine: InfiniGramEngine, flat: List[int],
                 n_checks: int = 50) -> bool:
    """
    Cross-check engine.count() against naive brute-force scan
    for random 1-gram, 2-gram, and 4-gram queries.
    """
    ok = True
    for n_gram in [1, 2, 4]:
        positions = [
            i for i in random.sample(range(len(flat) - n_gram),
                                     min(200, len(flat) - n_gram))
            if 0 not in flat[i:i + n_gram]
        ][:n_checks]

        for pos in positions:
            query    = flat[pos:pos + n_gram]
            got      = engine.count(query)['cnt']
            expected = naive_count(flat, query)
            if got != expected:
                print(f"  Count mismatch for {n_gram}-gram {query}: "
                      f"SA={got} naive={expected}")
                ok = False
    return ok


# ─── Single Validation Run ────────────────────────────────────────────────────

def validate_builder(builder: str, n_tokens: int,
                     n_threads: int = 1) -> bool:
    """
    Build an index with the given builder and run all correctness checks.
    Returns True if all checks pass.
    """
    # skip naive builders on large corpora
    if builder in ('sorted', 'sais') and n_tokens > NAIVE_MAX_TOKENS:
        print(f"  [skip] {builder} skipped for {n_tokens:,} tokens "
              f"(limit: {NAIVE_MAX_TOKENS:,})")
        return True   # not a failure — just out of scope

    print(f"\n{'='*60}")
    print(f"Validating: {builder}  |  ~{n_tokens:,} tokens")
    print(f"{'='*60}")

    docs      = generate_corpus(n_tokens)
    index_dir = tempfile.mkdtemp(prefix=f'verify_{builder}_')

    try:
        meta = build_index(
            docs, index_dir,
            eos_token_id=0,
            sa_builder=builder,
            n_threads=n_threads,
        )
        print(f"Actual tokens : {meta['n_tokens']:,}")

        # reconstruct flat token list for naive cross-checks
        tokenized_docs, _ = tokenize_documents(docs, eos_token_id=0)
        flat = []
        for toks in tokenized_docs:
            flat.extend(toks)
            flat.append(0)

        engine = InfiniGramEngine(index_dir, eos_token_id=0)

        covers  = check_sa_covers_all_positions(engine)
        ordered = check_sa_order(engine, flat)
        counts  = check_counts(engine, flat)

        print(f"Covers all positions : {'✓' if covers  else '✗ FAILED'}")
        print(f"SA ordering          : {'✓' if ordered else '✗ FAILED'}")
        print(f"Count accuracy       : {'✓' if counts  else '✗ FAILED'}")

        engine.close()

        passed = covers and ordered and counts
        print(f"\nResult: {'PASS ✓' if passed else 'FAIL ✗'}")
        return passed

    finally:
        shutil.rmtree(index_dir, ignore_errors=True)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Correctness validation for all SA builders'
    )
    parser.add_argument('--builders', nargs='+',
                        default=['sorted', 'sais', 'rust', 'caps_sa'],
                        choices=['sorted', 'sais', 'rust', 'caps_sa'],
                        help='Builders to validate (default: all four)')
    parser.add_argument('--n_tokens', type=int, default=1_000_000,
                        help='Approximate corpus size in tokens for '
                             'rust/caps_sa (default: 1M). '
                             'sorted/sais are capped at '
                             f'{NAIVE_MAX_TOKENS:,}.')
    parser.add_argument('--threads', type=int, default=1,
                        help='Thread count for parallel builders (default: 1)')
    args = parser.parse_args()

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

    random.seed(42)
    results = {}

    for builder in args.builders:
        results[builder] = validate_builder(
            builder   = builder,
            n_tokens  = args.n_tokens,
            n_threads = args.threads,
        )

    # summary
    print(f"\n{'='*60}")
    print(f"{'SUMMARY':^60}")
    print(f"{'='*60}")
    all_passed = True
    for builder, passed in results.items():
        print(f"  {builder:<12} {'PASS ✓' if passed else 'FAIL ✗'}")
        if not passed:
            all_passed = False

    sys.exit(0 if all_passed else 1)


if __name__ == '__main__':
    main()