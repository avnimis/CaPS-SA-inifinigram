"""
infini_gram.py
==============
Core infini-gram Python approximation: index builder + query engine.

On-disk index layout (matches infini-gram binary format):
  tokenized.bin  — little-endian uint16 tokens, EOS-separated documents
  table.bin      — suffix array as 5-byte little-endian byte-offset pointers
  offset.bin     — 8-byte little-endian int64 document start byte offsets

Suffix array builders (select via sa_builder= in build_index):
  'sorted'   — pure-Python O(N² log N) via sorted()          [naive_sa.py]
  'sais'     — pure-Python O(N log N) prefix-doubling SAIS   [naive_sa.py]
  'rust'     — rust_indexing binary (make-part → merge → concat)
  'caps_sa'  — CaPS-SA binary (CaPS_SA <input> <output>)

References:
  Liu et al. (2024) https://arxiv.org/abs/2401.17377
  github.com/liujch1998/infini-gram
"""

import os
import mmap
import shutil
import struct
import tempfile
import time
import subprocess
import numpy as np
from typing import List, Tuple, Dict, Optional

from naive_sa import build_sa_sorted, build_sa_sais


# ─── Constants (must match infini-gram's binary layout) ───────────────────────

BYTES_PER_TOKEN  = 2   # uint16 little-endian per token
BYTES_PER_PTR    = 5   # 5-byte SA pointer (handles up to 500B tokens)
BYTES_PER_OFFSET = 8   # int64 document byte offset

# ─── Default binary locations ─────────────────────────────────────────────────

_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
_RUST_BIN    = os.path.join(_PROJECT_DIR, 'rust_indexing')
_CAPS_BIN    = os.path.join(_PROJECT_DIR, 'CaPS-SA', 'bin', 'caps_sa')


# Tokenization
def tokenize_documents(
    documents: List[str],
    eos_token_id: int = 0,
) -> Tuple[List[List[int]], Dict[str, int]]:
    """
    Whitespace tokenizer → uint16 IDs. Token 0 is reserved for EOS.

    In production, replace with:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", ...)
    """
    vocab: Dict[str, int] = {}
    tokenized: List[List[int]] = []
    for doc in documents:
        ids = []
        for word in doc.lower().split():
            if word not in vocab:
                new_id = len(vocab) + 1  # reserve 0 for EOS
                if new_id > 65535:
                    raise ValueError("Vocab exceeds uint16; use a real tokenizer")
                vocab[word] = new_id
            ids.append(vocab[word])
        tokenized.append(ids)
    return tokenized, vocab


def _write_table(sa: List[int], path: str) -> None:
    """
    Write suffix array as 5-byte little-endian byte-offset pointers,
    matching infini-gram's table.{s} binary format.
    byte_offset = token_index × BYTES_PER_TOKEN
    """
    with open(path, 'wb') as f:
        buf = bytearray(BYTES_PER_PTR * len(sa))
        for i, tok_idx in enumerate(sa):
            byte_off = tok_idx * BYTES_PER_TOKEN
            packed   = struct.pack('<Q', byte_off)[:BYTES_PER_PTR]
            buf[i * BYTES_PER_PTR:(i + 1) * BYTES_PER_PTR] = packed
        f.write(buf)


def _build_sa_rust(
    tok_path: str,
    table_path: str,
    rust_bin: str,
    n_threads: int = 1,
) -> None:
    """
    Build the suffix array via the rust_indexing binary (3-step pipeline).

    Pipeline:  make-part → merge → concat
    Output:    table_path written with BYTES_PER_PTR-byte LE byte-offset
               entries, directly compatible with InfiniGramEngine.

    The binary reinterprets uint16 LE tokens as big-endian internally so
    that byte-level lexicographic order matches token-integer order.
    """
    ds_size = os.path.getsize(tok_path)
    if ds_size == 0:
        raise ValueError("tokenized.bin is empty; cannot build SA")

    base       = os.path.dirname(table_path)
    parts_dir  = os.path.join(base, '_rust_parts')
    merged_dir = os.path.join(base, '_rust_merged')
    os.makedirs(parts_dir,  exist_ok=True)
    os.makedirs(merged_dir, exist_ok=True)

    try:
        # Step 1: build a partial SA for the whole file (single chunk)
        subprocess.run([
            rust_bin, 'make-part',
            '--data-file',   tok_path,
            '--parts-dir',   parts_dir,
            '--start-byte',  '0',
            '--end-byte',    str(ds_size),
            '--ratio',       str(BYTES_PER_PTR),
            '--token-width', str(BYTES_PER_TOKEN),
        ], check=True)

        # Step 2: merge partial SAs (trivial with one chunk, but required)
        subprocess.run([
            rust_bin, 'merge',
            '--data-file',   tok_path,
            '--parts-dir',   parts_dir,
            '--merged-dir',  merged_dir,
            '--num-threads', str(n_threads),
            '--hacksize',    str(min(100_000, ds_size)),
            '--ratio',       str(BYTES_PER_PTR),
            '--token-width', str(BYTES_PER_TOKEN),
        ], check=True)

        # Step 3: concatenate thread outputs → table_path
        subprocess.run([
            rust_bin, 'concat',
            '--data-file',   tok_path,
            '--merged-dir',  merged_dir,
            '--merged-file', table_path,
            '--num-threads', str(n_threads),
            '--ratio',       str(BYTES_PER_PTR),
            '--token-width', str(BYTES_PER_TOKEN),
        ], check=True)
    finally:
        shutil.rmtree(parts_dir,  ignore_errors=True)
        shutil.rmtree(merged_dir, ignore_errors=True)

def _encode_for_caps(tokens: np.ndarray) -> bytes:
    """
    Encode each uint16 token as 8 bytes, order-preserving through
    CaPS-SA's DNA mapping at the BYTE level (low byte first, high byte second).
    Each byte is encoded as 4 pairs of 2 bits using enc_map.
    enc_map preserves order: A(00) < C(01) < G(11) < T(10)
    """
    enc_map = np.array([0x01, 0x03, 0x07, 0x05], dtype=np.uint8)
    
    # split each uint16 into low byte and high byte
    lo = (tokens & 0xFF).astype(np.uint8)
    hi = (tokens >> 8).astype(np.uint8)

    # extract all 4 bit-pairs from each byte via vectorized shifts
    # shape: (n_tokens, 2 bytes, 4 bit-pairs) → (n_tokens, 8)
    shifts = np.array([6, 4, 2, 0], dtype=np.uint8)
    lo_pairs = (lo[:, None] >> shifts) & 0x3   # (N, 4)
    hi_pairs = (hi[:, None] >> shifts) & 0x3   # (N, 4)

    # map bit-pairs to enc_map values and interleave [lo, hi] per token
    encoded = np.empty((len(tokens), 8), dtype=np.uint8)
    encoded[:, 0:4] = enc_map[lo_pairs]
    encoded[:, 4:8] = enc_map[hi_pairs]

    return encoded.ravel().tobytes()
 


def _build_sa_caps(
    tok_path: str,
    table_path: str,
    caps_bin: str,
    n_threads: int = 1,
) -> None:
    """
    Build the suffix array via a CaPS-SA binary.

    Expected binary interface::

        caps_bin <input_path> <output_path>

    The binary must write its output in CaPS_SA::Suffix_Array::dump() format:

        [8 bytes]   n    — std::size_t (uint64 LE), character count
        [nxw bytes] SA   — idx_t[n]: uint32 if n≤2³², else uint64; byte positions
        [nxw bytes] LCP  — idx_t[n]: same width; discarded here

    SA values are byte positions in tok_path.  Only token-aligned (even)
    positions are retained and written as BYTES_PER_PTR-byte LE byte-offset
    pointers in table_path.

    Note: the CaPS-SA binary in this project maps input bytes to {A,C,T,G}
    before constructing the SA (designed for DNA sequences).  For correct
    results over binary uint16 token data you need a CaPS-SA build without
    that character-mapping step in src/main.cpp.
    """
    env = os.environ.copy()
    env['PARLAY_NUM_THREADS'] = str(n_threads)

    fd_enc, enc_path   = tempfile.mkstemp(suffix='.caps_enc')
    fd_sa,  raw_sa_path = tempfile.mkstemp(suffix='.caps_sa')
    os.close(fd_enc)
    os.close(fd_sa)

    try:
        tokens = np.fromfile(tok_path, dtype='<u2')
        enc = _encode_for_caps(tokens)

        with open(enc_path, 'wb') as f:
            f.write(enc)

        subprocess.run([caps_bin, enc_path, raw_sa_path], env=env, check=True)

        with open(raw_sa_path, 'rb') as f:
            n      = struct.unpack('<Q', f.read(8))[0]
            sa_raw = np.frombuffer(f.read(n * 4), dtype='<u4')


        # keep only positions at token bound postitions
        sa_aligned = sa_raw[sa_raw % 8 == 0]

        with open(table_path, 'wb') as f:
            byte_offs = (sa_aligned.astype(np.uint64) // 8 * BYTES_PER_TOKEN).astype('<u8')
            raw8 = byte_offs.view(np.uint8).reshape(-1, 8)
            f.write(np.ascontiguousarray(raw8[:, :BYTES_PER_PTR]).tobytes())
    finally:
        for p in [enc_path, raw_sa_path]:
            if os.path.exists(p):
                os.remove(p)




# Index Builder

def build_index(
    documents: List[str],
    index_dir: str,
    eos_token_id: int = 0,
    sa_builder: str = 'sais',
    rust_bin: Optional[str] = None,
    caps_bin: Optional[str] = None,
    n_threads: int = 1,
) -> dict:
    """
    Build an infini-gram-compatible on-disk index.

    Produces three files matching infini-gram's shard layout:
      tokenized.bin  — all token IDs (uint16 LE) with EOS separators
      table.bin      — suffix array (5-byte LE byte-offset pointers)
      offset.bin     — document start byte offsets (int64 LE)

    Args:
        sa_builder:  'sorted' | 'sais' | 'rust' | 'caps_sa'
        rust_bin:    path to rust_indexing binary (default: ./rust_indexing)
        caps_bin:    path/name of CaPS-SA binary  (default: 'CaPS_SA' on PATH)
        n_threads:   parallelism for rust/caps_sa builders

    Returns metadata dict: vocab, token count, doc count, and build times.
    """
    os.makedirs(index_dir, exist_ok=True)
    tokenized_docs, vocab = tokenize_documents(documents, eos_token_id)

    # 1. Flatten corpus with EOS separators → tokenized.bin
    t0 = time.perf_counter()
    flat: List[int]        = []
    doc_offsets: List[int] = []   # byte offsets into tokenized.bin

    for toks in tokenized_docs:
        doc_offsets.append(len(flat) * BYTES_PER_TOKEN)
        flat.extend(toks)
        flat.append(eos_token_id)

    tok_arr = np.array(flat, dtype='<u2')
    tok_arr.tofile(os.path.join(index_dir, 'tokenized.bin'))
    t_tok = time.perf_counter() - t0

    # 2. Document offsets → offset.bin
    off_arr = np.array(doc_offsets, dtype='<i8')
    off_arr.tofile(os.path.join(index_dir, 'offset.bin'))

    # 3. Suffix array → table.bin
    tok_path   = os.path.join(index_dir, 'tokenized.bin')
    table_path = os.path.join(index_dir, 'table.bin')

    t0 = time.perf_counter()
    if sa_builder == 'sorted':
        _write_table(build_sa_sorted(flat), table_path)
    elif sa_builder == 'sais':
        _write_table(build_sa_sais(flat), table_path)
    elif sa_builder == 'rust':
        _build_sa_rust(tok_path, table_path,
                       rust_bin=rust_bin or _RUST_BIN,
                       n_threads=n_threads)
    elif sa_builder == 'caps_sa':
        _build_sa_caps(tok_path, table_path,
                       caps_bin=caps_bin or _CAPS_BIN,
                       n_threads=n_threads)
    else:
        raise ValueError(f"Unknown sa_builder: {sa_builder!r}")
    t_sa = time.perf_counter() - t0

    return {
        'vocab':      vocab,
        'n_tokens':   len(flat),
        'n_docs':     len(doc_offsets),
        't_tokenize': t_tok,
        't_sa':       t_sa,
    }


# Query Engine 
class InfiniGramEngine:
    """
    mmap-based query engine over an on-disk infini-gram-compatible index.

    Both tokenized.bin and table.bin are accessed via mmap (no full RAM
    copies), matching infini-gram's on-disk inference design.
    offset.bin is small and loaded fully into RAM as a numpy array.

    Public API mirrors infini-gram's Python package:
        engine.count(input_ids)
        engine.find(input_ids, max_results)
        engine.infgram_prob(prompt_ids, cont_id)
        engine.infgram_ntd(prompt_ids, max_support)
    """

    def __init__(self, index_dir: str, eos_token_id: int = 0):
        self.index_dir    = index_dir
        self.eos_token_id = eos_token_id

        tok_path = os.path.join(index_dir, 'tokenized.bin')
        sa_path  = os.path.join(index_dir, 'table.bin')
        off_path = os.path.join(index_dir, 'offset.bin')

        self._tok_f  = open(tok_path, 'rb')
        self._sa_f   = open(sa_path,  'rb')
        self._tok_mm = mmap.mmap(self._tok_f.fileno(), 0, access=mmap.ACCESS_READ)
        self._sa_mm  = mmap.mmap(self._sa_f.fileno(),  0, access=mmap.ACCESS_READ)

        self.n_tokens = os.path.getsize(tok_path) // BYTES_PER_TOKEN
        self.n_sa     = os.path.getsize(sa_path)  // BYTES_PER_PTR
        self._doc_offsets = np.fromfile(off_path, dtype='<i8')

        assert self.n_tokens == self.n_sa, (
            f"Corrupt index: token count {self.n_tokens} ≠ SA size {self.n_sa}"
        )

    def close(self):
        self._tok_mm.close(); self._tok_f.close()
        self._sa_mm.close();  self._sa_f.close()

    def __enter__(self):  return self
    def __exit__(self, *_): self.close()

    # Low-level mmap accessors
    def _tok(self, idx: int) -> int:
        off = idx * BYTES_PER_TOKEN
        return struct.unpack_from('<H', self._tok_mm, off)[0]

    def _sa(self, rank: int) -> int:
        off     = rank * BYTES_PER_PTR
        raw     = self._sa_mm[off: off + BYTES_PER_PTR]
        byte_off = struct.unpack('<Q', raw + b'\x00\x00\x00')[0]
        return byte_off // BYTES_PER_TOKEN

    # Binary search 
    # def _cmp(self, rank: int, query: List[int]) -> int:
    #     """
    #     Compare suffix at SA[rank] against query as a prefix.
    #     Returns -1 (suffix < query), 0 (match), 1 (suffix > query).
    #     """
    #     base = self._sa(rank)
    #     for j, q in enumerate(query):
    #         pos = base + j
    #         if pos >= self.n_tokens:
    #             return -1
    #         t = self._tok(pos)
    #         if t < q: return -1
    #         if t > q: return  1
    #     return 0
    def _cmp(self, rank: int, query: List[int]) -> int:
        base = self._sa(rank)
        for j, q in enumerate(query):
            pos = base + j
            if pos >= self.n_tokens:
                return -1
            # compare raw bytes, not integer values — matches rust_indexing sort order
            off     = pos * BYTES_PER_TOKEN
            t_bytes = bytes(self._tok_mm[off: off + BYTES_PER_TOKEN])
            q_bytes = struct.pack('<H', q)
            if t_bytes < q_bytes: return -1
            if t_bytes > q_bytes: return  1
        return 0

    def _lower_bound(self, query: List[int]) -> int:
        lo, hi = 0, self.n_sa
        while lo < hi:
            mid = (lo + hi) >> 1
            if self._cmp(mid, query) < 0: lo = mid + 1
            else:                          hi = mid
        return lo

    def _upper_bound(self, query: List[int]) -> int:
        lo, hi = 0, self.n_sa
        while lo < hi:
            mid = (lo + hi) >> 1
            if self._cmp(mid, query) <= 0: lo = mid + 1
            else:                           hi = mid
        return lo

    # Public query API
    def count(self, input_ids: List[int]) -> dict:
        """
        Count occurrences of input_ids in the corpus.
        O(|input_ids| · log N) via binary search on the suffix array.
        Returns: {'cnt': int, 'segment': (lo, hi)}
        """
        if not input_ids:
            return {'cnt': self.n_tokens, 'segment': (0, self.n_sa)}
        lo = self._lower_bound(input_ids)
        hi = self._upper_bound(input_ids)
        return {'cnt': hi - lo, 'segment': (lo, hi)}

    def find(self, input_ids: List[int], max_results: int = 10) -> dict:
        """
        Find all positions where input_ids occurs, with document context.
        Performs a binary search on offset.bin to recover the enclosing doc.
        Returns: {'cnt': int, 'results': [...], 'truncated': bool}
        Each result: {'token_offset', 'doc_ix', 'offset_within_doc'}
        """
        r = self.count(input_ids)
        lo, hi = r['segment']
        results = []
        for rank in range(lo, min(hi, lo + max_results)):
            tok_idx  = self._sa(rank)
            byte_off = tok_idx * BYTES_PER_TOKEN
            doc_idx  = int(np.searchsorted(self._doc_offsets, byte_off, side='right')) - 1
            doc_start = self._doc_offsets[doc_idx] // BYTES_PER_TOKEN
            results.append({
                'token_offset':      tok_idx,
                'doc_ix':            doc_idx,
                'offset_within_doc': tok_idx - doc_start,
            })
        return {'cnt': r['cnt'], 'results': results, 'truncated': r['cnt'] > max_results}

    def infgram_prob(self, prompt_ids: List[int], cont_id: int) -> dict:
        """
        ∞-gram probability of cont_id given prompt_ids.
        Backs off from the full prompt to shorter suffixes until finding
        a suffix with non-zero count.
        Returns: {'prompt_cnt', 'cont_cnt', 'prob', 'suffix_len'}
        """
        for sfx_len in range(len(prompt_ids), 0, -1):
            suffix     = prompt_ids[-sfx_len:]
            prompt_cnt = self.count(suffix)['cnt']
            if prompt_cnt > 0:
                cont_cnt = self.count(suffix + [cont_id])['cnt']
                return {
                    'prompt_cnt': prompt_cnt,
                    'cont_cnt':   cont_cnt,
                    'prob':       cont_cnt / prompt_cnt,
                    'suffix_len': sfx_len,
                }
        cont_cnt = self.count([cont_id])['cnt']
        return {
            'prompt_cnt': self.n_tokens,
            'cont_cnt':   cont_cnt,
            'prob':       cont_cnt / self.n_tokens if self.n_tokens else 0.0,
            'suffix_len': 0,
        }

    def infgram_ntd(
        self,
        prompt_ids: List[int],
        max_support: int = 10,
    ) -> dict:
        """
        ∞-gram next-token distribution given prompt_ids.
        Iterates over the matching SA segment to collect all observed
        next tokens; EOS tokens are excluded.
        Returns: {'prompt_cnt', 'result_by_token_id', 'suffix_len', 'approx'}
        """
        prompt_cnt  = 0
        best_suffix: List[int] = []
        best_len    = 0

        for sfx_len in range(len(prompt_ids), 0, -1):
            suffix = prompt_ids[-sfx_len:]
            c = self.count(suffix)['cnt']
            if c > 0:
                prompt_cnt  = c
                best_suffix = suffix
                best_len    = sfx_len
                break

        if prompt_cnt == 0:
            return {'prompt_cnt': 0, 'result_by_token_id': {}, 'suffix_len': 0}

        lo, hi = self.count(best_suffix)['segment']
        next_counts: Dict[int, int] = {}
        for rank in range(lo, hi):
            next_pos = self._sa(rank) + best_len
            if next_pos < self.n_tokens:
                t = self._tok(next_pos)
                if t != self.eos_token_id:
                    next_counts[t] = next_counts.get(t, 0) + 1

        top = sorted(next_counts.items(), key=lambda x: -x[1])[:max_support]
        result_by_token_id = {
            tok: {'cont_cnt': cnt, 'prob': cnt / prompt_cnt}
            for tok, cnt in top
        }
        return {
            'prompt_cnt':         prompt_cnt,
            'result_by_token_id': result_by_token_id,
            'suffix_len':         best_len,
            'approx':             len(next_counts) > max_support,
        }