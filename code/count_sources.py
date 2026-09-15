#!/usr/bin/env python3
"""
count_sources.py -- Classify the codewords of a clique by the nature of their
support, and report how the three kinds described in the paper are distributed:

  Source 1: linear codewords (trivial permutation at every support vector)
  Source 2: the support is a k-dimensional subspace of F_2^n, but the
            permutation assignment is nontrivial
  Source 3: the support is not a subspace

The script also reports, for reference, how many of the supports lie in one
fixed Desarguesian spread. That count turns out to be small (3, 1, 2 and 0 at
the four parameter sets of the main theorem), which is why the paper
classifies by subspace/nonsubspace rather than by membership in a spread.

A greedy lower bound on the number of pairwise disjoint subspace supports is
also printed. It is only a lower bound, and a weak one: for the exact maximum,
which the paper quotes, use max_disjoint_subspaces.py.

Usage (from the folder holding the clique files):
    python count_sources.py --clique clique_442.json   --n 4  --k 2
    python count_sources.py --clique clique_663.json   --n 6  --k 3
    python count_sources.py --clique clique_884.json   --n 8  --k 4
    python count_sources.py --clique clique_10105.json --n 10 --k 5

Uses only the standard library.
"""
import argparse
import json
import time
from collections import Counter

ap = argparse.ArgumentParser()
ap.add_argument("--clique", required=True)
ap.add_argument("--n", type=int, required=True)
ap.add_argument("--k", type=int, required=True)
a = ap.parse_args()

n, k = a.n, a.k
zero = tuple([0] * n)
idp = tuple(range(n))

# ------------------------------------------------------------------ loading
data = json.load(open(a.clique))
if isinstance(data, dict):
    for f in ("codes", "candidates", "clique", "pool"):
        if f in data:
            data = data[f]
            break
clique = [[(tuple(v), tuple(p)) for v, p in C] for C in data]
M = len(clique)
print(f"clique: {M} codewords of order {2**k} in U_{n}\n")


def support(C):
    return frozenset(v for v, p in C)


def is_subspace(S):
    Sset = set(S)
    return all(tuple(x ^ y for x, y in zip(u, v)) in Sset
               for u in Sset for v in Sset)


def is_linear(C):
    return all(p == idp for v, p in C)


# ------------------------------------------------- the three kinds of codeword
t0 = time.time()
sups = [support(C) for C in clique]
distinct = set(sups)
subsp = {s for s in distinct if is_subspace(s)}

n_src1 = sum(1 for C in clique if is_linear(C))
n_src2 = sum(1 for C in clique if support(C) in subsp and not is_linear(C))
n_src3 = sum(1 for C in clique if support(C) not in subsp)

print("codewords by kind")
print(f"  Source 1 (linear) ....................... {n_src1}")
print(f"  Source 2 (subspace support, nontrivial) . {n_src2}")
print(f"  Source 3 (nonlinear support) ............ {n_src3}")
assert n_src1 + n_src2 + n_src3 == M

print("\nsupports")
print(f"  distinct ................................ {len(distinct)}")
print(f"  subspaces ............................... {len(subsp)}")
print(f"  not subspaces ........................... {len(distinct)-len(subsp)}"
      f"  ({100*(len(distinct)-len(subsp))/len(distinct):.0f}%)")
print(f"  ({time.time()-t0:.0f}s)")


# ----------------------------------- one fixed Desarguesian spread, for reference
IRRED = {2: 0b111, 3: 0b1011, 4: 0b10011, 5: 0b100101, 6: 0b1000011}


def pmod(x, f):
    df = f.bit_length() - 1
    while x.bit_length() - 1 >= df:
        x ^= f << (x.bit_length() - 1 - df)
    return x


def pmul(x, y, f):
    r = 0
    while y:
        if y & 1:
            r ^= x
        y >>= 1
        x = pmod(x << 1, f)
    return pmod(r, f)


def desarguesian_spread(n, k):
    """The k-dimensional members of one Desarguesian spread of F_2^n (k | n)."""
    assert n % k == 0
    r, f, q = n // k, IRRED[k], 2 ** k

    def to_bits(coords):
        v = []
        for c in coords:
            v.extend((c >> t) & 1 for t in range(k))
        return tuple(v)

    points = []
    for lead in range(r):
        tail = r - lead - 1
        for mval in range(q ** tail):
            coords = [0] * lead + [1]
            mm = mval
            for _ in range(tail):
                coords.append(mm % q)
                mm //= q
            points.append(tuple(coords))
    return [frozenset(to_bits(tuple(pmul(lam, c, f) for c in pt))
                      for lam in range(q)) for pt in points]


if n % k == 0 and k in IRRED:
    spread = {frozenset(V) for V in desarguesian_spread(n, k)}
    in_spread = {s for s in subsp if s in spread}
    print("\nfor reference: supports lying in one fixed Desarguesian spread")
    print(f"  {len(in_spread)} of the {len(spread)} spread members occur as supports")
    print("  (a Desarguesian spread is not unique, so this count is relative")
    print("   to the particular spread built here)")
else:
    print("\n(k does not divide n, or k outside the tabulated irreducibles:")
    print(" no Desarguesian spread of this shape)")


# ------------------------- greedy lower bound on pairwise disjoint subspaces
chosen, used = [], set()
for s in sorted(subsp):
    core = s - {zero}
    if core & used:
        continue
    chosen.append(s)
    used |= core
full = (2 ** n - 1) // (2 ** k - 1) if n % k == 0 else None
print(f"\npairwise disjoint subspace supports (greedy LOWER bound): "
      f"{len(chosen)} of {len(subsp)}")
if full:
    print(f"  a full spread would have {full}")
print("  this greedy value is weak; run max_disjoint_subspaces.py for the")
print("  exact maximum, which is what the paper quotes")


# ------------------------------------------------------------- multiplicities
mult = Counter(sups)
rep = Counter(mult.values())
shared = sum(c for m, c in rep.items() if m >= 2)
print(f"\nsupports carrying several codewords: {shared}")
print(f"  multiplicity distribution: {dict(sorted(rep.items()))}")
