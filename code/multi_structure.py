#!/usr/bin/env python3
"""
multi_structure.py -- Examine the supports of a clique that carry several
pairwise disjoint propelinear structures.

For each such support the script reports how many codewords share it, whether
the support is a subspace of F_2^n, and its F_2-rank. The question it answers
is whether the multi-structure mechanism is confined to subspace supports
(where the setwise stabilizer in S_n is large, so there is more room for
distinct permutation assignments) or also occurs on nonlinear ones.

Usage (from the folder holding the clique files):
    python multi_structure.py --clique clique_442.json   --n 4  --k 2
    python multi_structure.py --clique clique_663.json   --n 6  --k 3
    python multi_structure.py --clique clique_884.json   --n 8  --k 4
    python multi_structure.py --clique clique_10105.json --n 10 --k 5

Uses only the standard library.
"""
import argparse
import json
from collections import Counter, defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--clique", required=True)
ap.add_argument("--n", type=int, required=True)
ap.add_argument("--k", type=int, required=True)
a = ap.parse_args()

n, k = a.n, a.k

data = json.load(open(a.clique))
if isinstance(data, dict):
    for f in ("codes", "candidates", "clique", "pool"):
        if f in data:
            data = data[f]
            break
clique = [[(tuple(v), tuple(p)) for v, p in C] for C in data]
print(f"clique: {len(clique)} codewords of order {2**k} in U_{n}\n")


def support(C):
    return frozenset(v for v, p in C)


def is_subspace(S):
    Sset = set(S)
    return all(tuple(x ^ y for x, y in zip(u, v)) in Sset
               for u in Sset for v in Sset)


def rank(S):
    basis = []
    for v in S:
        w = list(v)
        for b in basis:
            i = next(j for j, x in enumerate(b) if x)
            if w[i]:
                w = [(w[j] + b[j]) % 2 for j in range(n)]
        if any(w):
            basis.append(w)
    return len(basis)


by_support = defaultdict(list)
for idx, C in enumerate(clique):
    by_support[support(C)].append(idx)

multi = {s: idxs for s, idxs in by_support.items() if len(idxs) >= 2}
if not multi:
    print("no support carries more than one codeword")
    raise SystemExit(0)

print(f"{len(multi)} supports carry several codewords\n")
print(f"{'structures':>11}  {'subspace?':>10}  {'rank':>5}  count")
rows = Counter()
for s, idxs in multi.items():
    rows[(len(idxs), is_subspace(s), rank(s))] += 1
for (m, sub, r), c in sorted(rows.items(), key=lambda t: (-t[0][0], t[0][1])):
    print(f"{m:>11}  {'yes' if sub else 'no':>10}  {r:>5}  {c}")

# summary: extra codewords attributable to each kind of support
extra_sub = sum(len(i) - 1 for s, i in multi.items() if is_subspace(s))
extra_non = sum(len(i) - 1 for s, i in multi.items() if not is_subspace(s))
n_sub = sum(1 for s in multi if is_subspace(s))
print(f"\nmulti-structure supports that are subspaces ..... {n_sub}")
print(f"                          that are not .......... {len(multi)-n_sub}")
print(f"extra codewords from subspace supports .......... {extra_sub}")
print(f"                  from nonlinear supports ....... {extra_non}")

top = max(multi.items(), key=lambda t: len(t[1]))
s, idxs = top
print(f"\nlargest: {len(idxs)} structures on one support, "
      f"{'a subspace' if is_subspace(s) else 'NOT a subspace'}, rank {rank(s)}")
