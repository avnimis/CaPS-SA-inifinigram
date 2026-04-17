"""
naive_sa.py
===========
Pure-Python suffix array builders (naive / baseline implementations).

Two variants are provided:
  build_sa_sorted  — O(N² log N) via Python's built-in sorted()
  build_sa_sais    — O(N log² N) prefix-doubling (Manber & Myers style)

Both sort by raw little-endian bytes, matching the sort order used by
rust_indexing and CaPS-SA, so all builders produce compatible suffix
arrays that can be queried by the same InfiniGramEngine._cmp() logic.

These are correct for any token sequence but only practical for small
corpora (~50k tokens for sorted, ~500k for sais). For production use
the rust or caps_sa builders in infini_gram.py instead.
"""
v
import struct
from typing import List


def _tok_bytes(tokens: List[int], i: int) -> bytes:
    """
    Return the byte representation of the suffix starting at token index i,
    encoding each uint16 token as 2 little-endian bytes.
    Matches the byte-level sort order used by rust_indexing and CaPS-SA.
    """
    return b''.join(struct.pack('<H', t) for t in tokens[i:])


def build_sa_sorted(tokens: List[int]) -> List[int]:
    """
    O(N² log N) suffix array via Python's sorted().
    Sorts by raw little-endian bytes to match rust_indexing/CaPS-SA order.
    Practical only for corpora < ~50k tokens.
    """
    return sorted(range(len(tokens)), key=lambda i: _tok_bytes(tokens, i))


def build_sa_sais(tokens: List[int]) -> List[int]:
    """
    O(N log² N) suffix array via prefix-doubling (Manber & Myers style).
    Sorts by raw little-endian bytes to match rust_indexing/CaPS-SA order.
    Practical for corpora up to ~500k tokens.

    For production (GB+) use the rust or caps_sa builders in infini_gram.py.
    """
    N = len(tokens)
    if N == 0:
        return []

    # Initial ranks are raw little-endian byte values of each token.
    # Using the 2-byte LE representation as a uint16 big-endian integer
    # gives the correct byte-level ordering:
    #   struct.pack('<H', v) interpreted as big-endian = (v & 0xFF) << 8 | (v >> 8)
    def le_rank(v: int) -> int:
        return ((v & 0xFF) << 8) | ((v >> 8) & 0xFF)

    rank = [le_rank(t) for t in tokens] + [0]   # sentinel at position N
    sa   = list(range(N))
    tmp  = [0] * N

    k = 1
    while k < N:
        def key(i, _k=k, _r=rank):
            return (_r[i], _r[i + _k] if i + _k < len(_r) else -1)

        sa.sort(key=key)

        tmp[sa[0]] = 0
        for i in range(1, N):
            tmp[sa[i]] = tmp[sa[i - 1]]
            if key(sa[i]) != key(sa[i - 1]):
                tmp[sa[i]] += 1
        rank = tmp[:] + [0]

        if rank[sa[-1]] == N - 1:
            break   # all ranks unique — done
        k *= 2

    return sa