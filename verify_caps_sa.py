# verify_caps_sa.py
"""
Quick end-to-end correctness check for the CaPS-SA builder.

Validates against the 'sorted' baseline (pure Python, known correct):
  1. SA coverage    — every token position appears exactly once
  2. SA ordering    — spot-check byte-level lexicographic order
  3. Count accuracy — n-gram counts match for 1-, 2-, and 4-grams

The CaPS-SA binary must be built from the modified source (no DNA
remapping, dump_sa_only output) before running this script.
"""

import os
import struct
import random
import shutil
import tempfile
import numpy as np

from infini_gram import build_index, tokenize_documents, InfiniGramEngine, _CAPS_BIN

SEED = 42
random.seed(SEED)
rng = np.random.default_rng(SEED)


# ── Corpus ────────────────────────────────────────────────────────────────────

def make_corpus(n_docs: int = 20, doc_len: int = 500, vocab: int = 200) -> list[str]:
    """Diverse synthetic corpus with Zipf token frequencies."""
    weights = 1.0 / np.arange(1, vocab + 1)
    weights /= weights.sum()
    docs = []
    for _ in range(n_docs):
        toks = rng.choice(np.arange(1, vocab + 1), size=doc_len, p=weights)
        docs.append(' '.join(str(t) for t in toks))
    return docs


# ── Checks ────────────────────────────────────────────────────────────────────

def check_coverage(engine: InfiniGramEngine) -> bool:
    """Every token position must appear exactly once in the SA."""
    positions = [engine._sa(r) for r in range(engine.n_sa)]
    ok = sorted(positions) == list(range(engine.n_tokens))
    if not ok:
        missing = set(range(engine.n_tokens)) - set(positions)
        print(f"  Missing positions (first 5): {list(missing)[:5]}")
    return ok


def check_ordering(engine: InfiniGramEngine, flat: list[int],
                   sample: int = 500) -> bool:
    """Spot-check that adjacent SA entries are in byte-level lex order."""
    ranks = random.sample(range(engine.n_sa - 1), min(sample, engine.n_sa - 1))
    for rank in ranks:
        a = engine._sa(rank)
        b = engine._sa(rank + 1)
        sa = b''.join(struct.pack('<H', t) for t in flat[a:a + 8])
        sb = b''.join(struct.pack('<H', t) for t in flat[b:b + 8])
        if sa > sb:
            print(f"  Out of order at rank {rank}: {flat[a:a+4]} > {flat[b:b+4]}")
            return False
    return True


def check_counts(caps: InfiniGramEngine, baseline: InfiniGramEngine,
                 flat: list[int], n_checks: int = 40) -> bool:
    """Compare caps_sa counts against sorted baseline for 1-, 2-, 4-grams."""
    ok = True
    for n_gram in [1, 2, 4]:
        positions = [
            i for i in random.sample(
                range(len(flat) - n_gram),
                min(300, len(flat) - n_gram),
            )
            if 0 not in flat[i:i + n_gram]
        ][:n_checks]

        for pos in positions:
            query = flat[pos:pos + n_gram]
            got      = caps.count(query)['cnt']
            expected = baseline.count(query)['cnt']
            if got != expected:
                print(f"  {n_gram}-gram {query}: caps={got} sorted={expected}")
                ok = False
    return ok


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    if not os.path.isfile(_CAPS_BIN):
        print(f"ERROR: CaPS-SA binary not found at {_CAPS_BIN}")
        print("Build it with: cd CaPS-SA && mkdir build && cd build && cmake .. && make install")
        raise SystemExit(1)

    docs = make_corpus()
    tokenized_docs, _ = tokenize_documents(docs, eos_token_id=0)
    flat: list[int] = []
    for toks in tokenized_docs:
        flat.extend(toks)
        flat.append(0)

    caps_dir   = tempfile.mkdtemp(prefix='caps_verify_')
    sorted_dir = tempfile.mkdtemp(prefix='sorted_verify_')

    try:
        build_index(docs, caps_dir,   sa_builder='caps_sa')
        build_index(docs, sorted_dir, sa_builder='sorted')

        eng_caps   = InfiniGramEngine(caps_dir)
        eng_sorted = InfiniGramEngine(sorted_dir)

        print(f"Tokens: {eng_caps.n_tokens:,}  SA entries: {eng_caps.n_sa:,}")

        coverage = check_coverage(eng_caps)
        ordering = check_ordering(eng_caps, flat)
        counts   = check_counts(eng_caps, eng_sorted, flat)

        eng_caps.close()
        eng_sorted.close()

        print(f"Coverage  : {'PASS' if coverage else 'FAIL'}")
        print(f"Ordering  : {'PASS' if ordering else 'FAIL'}")
        print(f"Counts    : {'PASS' if counts   else 'FAIL'}")

        if coverage and ordering and counts:
            print("\nCaPS-SA verification PASSED ✓")
        else:
            print("\nCaPS-SA verification FAILED ✗")
            raise SystemExit(1)

    finally:
        shutil.rmtree(caps_dir,   ignore_errors=True)
        shutil.rmtree(sorted_dir, ignore_errors=True)


if __name__ == '__main__':
    main()
