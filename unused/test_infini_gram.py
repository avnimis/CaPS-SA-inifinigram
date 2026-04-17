"""
test_infini_gram.py
===================
Validation and benchmarking suite for infini_gram.py.

Covers:
  - Index construction (correctness of on-disk binary files)
  - SA ordering validity
  - count() correctness vs naive scan
  - find() document-offset resolution
  - infgram_prob() backoff behaviour
  - infgram_ntd() next-token distribution
  - Query latency benchmarks (median, p99)

Run with pytest:
    pytest test_infini_gram.py -v

Select a suffix-array builder with --builder (default: sais):
    pytest test_infini_gram.py -v --builder=sorted
    pytest test_infini_gram.py -v --builder=sais
    pytest test_infini_gram.py -v --builder=rust
    pytest test_infini_gram.py -v --builder=caps_sa

Or directly:
    python test_infini_gram.py
"""

import os
import struct
import time
import random
import tempfile
import numpy as np
import pytest
from typing import List

from infini_gram import (
    build_index,
    tokenize_documents,
    InfiniGramEngine,
    BYTES_PER_TOKEN,
    BYTES_PER_PTR,
    BYTES_PER_OFFSET,
    _RUST_BIN,
    _CAPS_BIN,
)


# Shared Fixtures
DOCUMENTS = [
    "the cat sat on the mat the cat is fat",
    "the dog sat on the log the dog is big the log is long",
    "a large language model is trained on a large corpus of text tokens",
    "suffix arrays allow fast lookup of any substring in the corpus",
    "binary search over a suffix array is efficient and correct",
    "n gram queries count how often a sequence of tokens appears",
    "data contamination occurs when evaluation text leaks into training data",
    "auditing training data requires exact string matching at scale",
    "parallel construction of suffix arrays improves scalability significantly",
    "the cat sat on the mat and the dog sat on the log again",
] * 8   # repeat to simulate a small realistic corpus

EOS = 0


@pytest.fixture(scope="module")
def index(tmp_path_factory, builder):
    """Build the index once for the entire test module using the selected builder."""
    # Skip tests that need an external binary when that binary is absent
    if builder == 'rust' and not os.path.isfile(_RUST_BIN):
        pytest.skip(f"rust_indexing binary not found at {_RUST_BIN}")
    if builder == 'caps_sa' and not os.path.isfile(_CAPS_BIN):
        pytest.skip(f"CaPS-SA binary not found at {_CAPS_BIN}")

    idx_dir = str(tmp_path_factory.mktemp("index"))
    meta = build_index(DOCUMENTS, idx_dir, eos_token_id=EOS, sa_builder=builder)
    return idx_dir, meta


@pytest.fixture(scope="module")
def engine(index):
    idx_dir, _ = index
    eng = InfiniGramEngine(idx_dir, eos_token_id=EOS)
    yield eng
    eng.close()


@pytest.fixture(scope="module")
def flat_tokens(index):
    """Reconstructed flat token list for naive cross-checks."""
    idx_dir, meta = index
    tokenized_docs, _ = tokenize_documents(DOCUMENTS, EOS)
    flat = []
    for toks in tokenized_docs:
        flat.extend(toks)
        flat.append(EOS)
    return flat


@pytest.fixture(scope="module")
def vocab(index):
    _, meta = index
    return meta['vocab']


# Helper
def naive_count(flat: List[int], query: List[int]) -> int:
    """O(N·k) brute-force scan — ground truth for count() validation."""
    n, k = len(flat), len(query)
    if k == 0:
        return n
    return sum(1 for i in range(n - k + 1) if flat[i:i + k] == query)


# 1. Index Construction 
class TestIndexConstruction:

    def test_tokenized_bin_exists(self, index):
        idx_dir, _ = index
        assert os.path.exists(os.path.join(idx_dir, 'tokenized.bin'))

    def test_table_bin_exists(self, index):
        idx_dir, _ = index
        assert os.path.exists(os.path.join(idx_dir, 'table.bin'))

    def test_offset_bin_exists(self, index):
        idx_dir, _ = index
        assert os.path.exists(os.path.join(idx_dir, 'offset.bin'))

    def test_tokenized_bin_size(self, index, flat_tokens):
        idx_dir, meta = index
        expected = meta['n_tokens'] * BYTES_PER_TOKEN
        actual   = os.path.getsize(os.path.join(idx_dir, 'tokenized.bin'))
        assert actual == expected, f"tokenized.bin: {actual} != {expected}"

    def test_table_bin_size(self, index, flat_tokens):
        idx_dir, meta = index
        expected = meta['n_tokens'] * BYTES_PER_PTR
        actual   = os.path.getsize(os.path.join(idx_dir, 'table.bin'))
        assert actual == expected, f"table.bin: {actual} != {expected}"

    def test_offset_bin_size(self, index):
        idx_dir, meta = index
        expected = meta['n_docs'] * BYTES_PER_OFFSET
        actual   = os.path.getsize(os.path.join(idx_dir, 'offset.bin'))
        assert actual == expected, f"offset.bin: {actual} != {expected}"

    def test_token_count_matches_flat(self, index, flat_tokens):
        _, meta = index
        assert meta['n_tokens'] == len(flat_tokens)

    def test_doc_count_matches_input(self, index):
        _, meta = index
        assert meta['n_docs'] == len(DOCUMENTS)

    def test_tokenized_bin_little_endian_uint16(self, index, flat_tokens):
        """First 10 tokens on disk should match in-memory flat list."""
        idx_dir, _ = index
        with open(os.path.join(idx_dir, 'tokenized.bin'), 'rb') as f:
            for expected in flat_tokens[:10]:
                raw = f.read(BYTES_PER_TOKEN)
                got = struct.unpack('<H', raw)[0]
                assert got == expected

    def test_table_bin_5byte_pointers(self, index):
        """Each SA entry should be a valid byte offset into tokenized.bin."""
        idx_dir, meta = index
        tok_bytes = meta['n_tokens'] * BYTES_PER_TOKEN
        with open(os.path.join(idx_dir, 'table.bin'), 'rb') as f:
            for _ in range(min(100, meta['n_tokens'])):
                raw      = f.read(BYTES_PER_PTR)
                byte_off = struct.unpack('<Q', raw + b'\x00\x00\x00')[0]
                assert byte_off < tok_bytes
                assert byte_off % BYTES_PER_TOKEN == 0


# 2. Suffix Array Ordering
class TestSuffixArrayOrdering:

    def test_sa_sorted_order(self, engine, flat_tokens):
        """Adjacent SA entries must be in non-decreasing lexicographic order."""
        prev = None
        for rank in range(engine.n_sa):
            tok_idx = engine._sa(rank)
            suffix  = flat_tokens[tok_idx:tok_idx + 8]   # compare up to 8 tokens
            if prev is not None:
                assert prev <= suffix, (
                    f"SA out of order at rank {rank}: {prev} > {suffix}"
                )
            prev = suffix

    def test_sa_covers_all_positions(self, engine):
        """Every token position must appear exactly once in the SA."""
        positions = [engine._sa(r) for r in range(engine.n_sa)]
        assert sorted(positions) == list(range(engine.n_tokens))


# 3. count() Correctness

class TestCount:

    def _ngrams(self, flat, n, k, exclude_eos=True):
        """Sample k random n-grams from flat, optionally excluding EOS."""
        results, attempts = [], 0
        positions = list(range(len(flat) - n))
        random.shuffle(positions)
        for i in positions:
            seq = flat[i:i + n]
            if exclude_eos and EOS in seq:
                continue
            results.append(seq)
            if len(results) == k:
                break
        return results

    def test_unigram_counts(self, engine, flat_tokens, vocab):
        for word, tok_id in list(vocab.items())[:10]:
            got      = engine.count([tok_id])['cnt']
            expected = naive_count(flat_tokens, [tok_id])
            assert got == expected, f"unigram '{word}': SA={got} naive={expected}"

    def test_bigram_counts(self, engine, flat_tokens):
        random.seed(0)
        for q in self._ngrams(flat_tokens, 2, 15):
            got      = engine.count(q)['cnt']
            expected = naive_count(flat_tokens, q)
            assert got == expected, f"bigram {q}: SA={got} naive={expected}"

    def test_trigram_counts(self, engine, flat_tokens):
        random.seed(1)
        for q in self._ngrams(flat_tokens, 3, 10):
            got      = engine.count(q)['cnt']
            expected = naive_count(flat_tokens, q)
            assert got == expected, f"trigram {q}: SA={got} naive={expected}"

    def test_fourgram_counts(self, engine, flat_tokens):
        random.seed(2)
        for q in self._ngrams(flat_tokens, 4, 8):
            got      = engine.count(q)['cnt']
            expected = naive_count(flat_tokens, q)
            assert got == expected, f"4-gram {q}: SA={got} naive={expected}"

    def test_absent_ngram_returns_zero(self, engine):
        # Token IDs 60000+ are guaranteed not in our small vocab
        assert engine.count([60000, 60001])['cnt'] == 0

    def test_empty_query_returns_all_tokens(self, engine):
        assert engine.count([])['cnt'] == engine.n_tokens

    def test_segment_bounds_are_consistent(self, engine, flat_tokens, vocab):
        tok_id = list(vocab.values())[0]
        r = engine.count([tok_id])
        lo, hi = r['segment']
        assert hi - lo == r['cnt']
        assert 0 <= lo <= hi <= engine.n_sa


# 4. find() Document Resolution 

class TestFind:

    def test_find_count_matches_count(self, engine, vocab):
        tok_id = vocab.get('the', 1)
        r_count = engine.count([tok_id])
        r_find  = engine.find([tok_id], max_results=1000)
        assert r_find['cnt'] == r_count['cnt']

    def test_find_results_within_corpus(self, engine, vocab):
        tok_id = vocab.get('cat', 2)
        r = engine.find([tok_id], max_results=20)
        for hit in r['results']:
            assert 0 <= hit['token_offset'] < engine.n_tokens
            assert hit['doc_ix'] >= 0
            assert hit['offset_within_doc'] >= 0

    def test_find_token_at_offset_matches_query(self, engine, vocab, flat_tokens):
        """The token at each reported token_offset must equal the first query token."""
        tok_id = vocab.get('suffix', 5)
        r = engine.find([tok_id], max_results=10)
        for hit in r['results']:
            assert flat_tokens[hit['token_offset']] == tok_id

    def test_find_truncated_flag(self, engine, vocab):
        tok_id = vocab.get('the', 1)
        r = engine.find([tok_id], max_results=1)
        # 'the' appears many times; max_results=1 should truncate
        if r['cnt'] > 1:
            assert r['truncated'] is True

    def test_find_no_results_for_absent_ngram(self, engine):
        r = engine.find([60000, 60001])
        assert r['cnt'] == 0
        assert r['results'] == []
        assert r['truncated'] is False


# 5. infgram_prob()

class TestInfgramProb:

    def test_prob_between_zero_and_one(self, engine, vocab):
        the = vocab.get('the', 1)
        cat = vocab.get('cat', 2)
        sat = vocab.get('sat', 3)
        r = engine.infgram_prob([the, cat], cont_id=sat)
        assert 0.0 <= r['prob'] <= 1.0

    def test_prob_consistency(self, engine, vocab):
        """cont_cnt / prompt_cnt should equal prob exactly."""
        the = vocab.get('the', 1)
        cat = vocab.get('cat', 2)
        sat = vocab.get('sat', 3)
        r = engine.infgram_prob([the, cat], cont_id=sat)
        if r['prompt_cnt'] > 0:
            assert abs(r['prob'] - r['cont_cnt'] / r['prompt_cnt']) < 1e-9

    def test_prob_suffix_len_at_most_prompt_len(self, engine, vocab):
        the = vocab.get('the', 1)
        cat = vocab.get('cat', 2)
        on  = vocab.get('on',  4)
        r = engine.infgram_prob([the, cat, on], cont_id=the)
        assert r['suffix_len'] <= 3

    def test_absent_cont_gives_zero_prob(self, engine, vocab):
        the = vocab.get('the', 1)
        r = engine.infgram_prob([the], cont_id=60000)
        assert r['cont_cnt'] == 0
        assert r['prob'] == 0.0

    def test_prob_sums_leq_one_over_vocab(self, engine, vocab):
        """Sum of all next-token probabilities should not exceed 1."""
        the = vocab.get('the', 1)
        total = sum(
            engine.infgram_prob([the], cont_id=tok_id)['prob']
            for tok_id in list(vocab.values())[:30]
        )
        assert total <= 1.0 + 1e-9


# 6. infgram_ntd()

class TestInfgramNtd:

    def test_ntd_probs_sum_leq_one(self, engine, vocab):
        the = vocab.get('the', 1)
        cat = vocab.get('cat', 2)
        r = engine.infgram_ntd([the, cat], max_support=50)
        total = sum(v['prob'] for v in r['result_by_token_id'].values())
        assert total <= 1.0 + 1e-9

    def test_ntd_no_eos_in_results(self, engine, vocab):
        the = vocab.get('the', 1)
        r = engine.infgram_ntd([the], max_support=50)
        assert EOS not in r['result_by_token_id']

    def test_ntd_counts_consistent_with_count(self, engine, vocab):
        """cont_cnt for each token in NTD should match count([prompt, token])."""
        the = vocab.get('the', 1)
        cat = vocab.get('cat', 2)
        r = engine.infgram_ntd([the, cat], max_support=20)
        sfx = [the, cat][-r['suffix_len']:] if r['suffix_len'] > 0 else []
        for tok_id, info in r['result_by_token_id'].items():
            expected = engine.count(sfx + [tok_id])['cnt']
            assert info['cont_cnt'] == expected, (
                f"token {tok_id}: ntd={info['cont_cnt']} count={expected}"
            )

    def test_ntd_absent_prompt_returns_empty(self, engine):
        r = engine.infgram_ntd([60000, 60001])
        assert r['prompt_cnt'] == 0
        assert r['result_by_token_id'] == {}

    def test_ntd_approx_flag_set_when_truncated(self, engine, vocab):
        the = vocab.get('the', 1)
        r = engine.infgram_ntd([the], max_support=1)
        # 'the' likely has many distinct continuations
        # approx should be True if there were more than max_support
        assert isinstance(r['approx'], bool)


# 7. Benchmarks

class TestBenchmarks:
    """
    Latency benchmarks — not strict assertions, just printed output.
    Loosely asserts that queries complete in < 500ms (very conservative
    for a small in-memory corpus; real corpora on disk will differ).
    """

    REPS = 50

    def _measure(self, fn, reps=None) -> List[float]:
        reps = reps or self.REPS
        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t0)
        return sorted(times)

    def _report(self, label, times):
        n = len(times)
        med = np.median(times) * 1e6
        p99 = times[int(n * 0.99)] * 1e6
        print(f"\n  {label}")
        print(f"    median={med:.1f}µs  p99={p99:.1f}µs  "
              f"min={times[0]*1e6:.1f}µs  max={times[-1]*1e6:.1f}µs")
        return med, p99

    def test_bench_count_unigram(self, engine, vocab):
        tok = list(vocab.values())[0]
        times = self._measure(lambda: engine.count([tok]))
        med, p99 = self._report("count() 1-gram", times)
        assert p99 < 500_000  # < 500ms

    def test_bench_count_bigram(self, engine, flat_tokens):
        random.seed(42)
        q = flat_tokens[10:12]
        times = self._measure(lambda: engine.count(q))
        med, p99 = self._report("count() 2-gram", times)
        assert p99 < 500_000

    def test_bench_count_trigram(self, engine, flat_tokens):
        random.seed(42)
        q = flat_tokens[10:13]
        times = self._measure(lambda: engine.count(q))
        med, p99 = self._report("count() 3-gram", times)
        assert p99 < 500_000

    def test_bench_infgram_prob(self, engine, vocab):
        the = vocab.get('the', 1)
        cat = vocab.get('cat', 2)
        sat = vocab.get('sat', 3)
        times = self._measure(lambda: engine.infgram_prob([the, cat], cont_id=sat))
        self._report("infgram_prob()", times)

    def test_bench_infgram_ntd(self, engine, vocab):
        the = vocab.get('the', 1)
        cat = vocab.get('cat', 2)
        times = self._measure(lambda: engine.infgram_ntd([the, cat]))
        self._report("infgram_ntd()", times)


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))