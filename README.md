# Scaling LLM Dataset Indexing with Parallel Suffix Arrays

CS4810 Final Project — Avni Mishra, Anh Nguyen

## Overview

This project extends the [infini-gram](https://arxiv.org/abs/2401.17377) system by replacing its single-threaded suffix array construction backend with [CaPS-SA](https://github.com/jamshed/CaPS-SA), a parallel suffix array construction algorithm. The goal is to accelerate index construction over large tokenized corpora while preserving exact n-gram query functionality.

LLMs are trained on massive datasets, but once training is complete the data is inaccessible - compressed into model weights. Suffix arrays enable exact n-gram lookup over tokenized corpora, which is useful for training data auditing, benchmark contamination detection, and transparency research. The bottleneck at scale is suffix array construction, which this project addresses through parallelism.

## Features

- Four interchangeable suffix array builders:
  - `sorted` — pure Python O(N² log N), baseline
  - `sais` — pure Python O(N log² N) prefix-doubling, baseline
  - `rust` — Rust-based SAIS from infini-gram
  - `caps_sa` — CaPS-SA parallel construction
- Exact n-gram count queries via binary search
- Next-token probability estimation with backoff
- Full correctness validation suite across all builders
- Benchmarking framework for construction time and query latency

## Project Structure

```
.
├── infini_gram.py          # Core engine: index building and query interface
├── naive_sa.py             # Pure-Python suffix array builders (sorted, sais)
├── test_infini_gram.py     # Unit test suite
├── conftest.py             # pytest config — --builder flag
├── verify_builders.py      # Correctness validation across all builders
└── plot_benchmarks.py      # Benchmark plotting scripts
```

## Requirements

```
Python 3.9+
numpy
pandas
matplotlib
```

Install dependencies:
```bash
pip install numpy pandas matplotlib
```

The `rust` and `caps_sa` builders also require their respective compiled binaries to be available on your system path.

## Usage

### Building an index

```python
from infini_gram import build_index, InfiniGramEngine

docs = ["the quick brown fox", "the fox jumped over the dog"]

meta = build_index(
    docs,
    index_dir="./my_index",
    eos_token_id=0,
    sa_builder="caps_sa",   # or "rust", "sais", "sorted"
    n_threads=4,
)

engine = InfiniGramEngine("./my_index", eos_token_id=0)
```

### Querying

```python
# Count occurrences of an n-gram
result = engine.count([101, 202])
print(result["cnt"])

# Next-token probability given context
result = engine.prob(context=[101, 202], next_token=303)
print(result["prob"])

engine.close()
```

### Running tests

```bash
# Run against a specific builder
pytest test_infini_gram.py -v --builder=caps_sa
pytest test_infini_gram.py -v --builder=rust
pytest test_infini_gram.py -v --builder=sais
```

### Validating correctness

```bash
# Validate all builders
python verify_builders.py

# Validate specific builders
python verify_builders.py --builders rust caps_sa

# Custom corpus size and thread count
python verify_builders.py --n_tokens 1000000 --threads 4
```

## Results Summary

Benchmarks were run on Northeastern's Explorer cluster on synthetic Zipf-distributed corpora.

- At 1 thread, CaPS-SA SA construction is ~3.7–3.9x slower than Rust due to parallelism overhead
- At 4 threads, CaPS-SA nearly matches Rust (within 2–8% on total construction time)
- Speedup plateaus beyond 4 threads for both builders
- Query latency is nearly identical across all builders (~100–180ms median at 600M tokens)
- SA construction accounts for 55–87% of total build time, confirming it as the primary bottleneck

## References

1. Liu et al. (2024). Infini-gram: Scaling Unbounded n-gram Language Models to a Trillion Tokens. arXiv:2401.17377.
2. Khan, J., Rubel, T., Dhulipala, L., Molloy, E., & Patro, R. (2023). Fast, Parallel, and Cache-Friendly Suffix Array Construction. WABI 2023. https://doi.org/10.4230/LIPIcs.WABI.2023.16
