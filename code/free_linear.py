#!/usr/bin/env python3
"""
free_linear.py -- Can any linear codeword be added to a clique at d = 2k?

A linear codeword {(v, id) : v in V} consists of translations only, so it is
compatible with every codeword of the clique exactly when none of the nonzero
vectors of V appears as a translation (v, id) inside the clique. The script
collects the nonzero translations left free by the clique and searches, by
backtracking, for a k-dimensional subspace all of whose nonzero vectors are
free. This avoids enumerating the Gaussian binomial, which is about 1.1e8 at
(n, k) = (10, 5).

    python free_linear.py --clique clique_10105.json --n 10 --k 5
"""
import argparse
import json

ap = argparse.ArgumentParser()
ap.add_argument("--clique", required=True)
ap.add_argument("--n", type=int, required=True)
ap.add_argument("--k", type=int, required=True)
a = ap.parse_args()
n, k = a.n, a.k

clique = json.load(open(a.clique))
idp = list(range(n))
zero = tuple([0] * n)
used = {tuple(v) for H in clique for v, p in H if p == idp and any(v)}
nonzero = [tuple((x >> (n - 1 - i)) & 1 for i in range(n)) for x in range(1, 2 ** n)]
free = [v for v in nonzero if v not in used]
allowed = set(free) | {zero}
print(f"{len(clique)} codewords; {len(used)} of {2**n - 1} nonzero translations "
      f"used, {len(free)} free")


def add(u, w):
    return tuple((u[i] + w[i]) % 2 for i in range(n))


def search(span, basis, start):
    """Extend span one free vector at a time, keeping it inside the free set."""
    if len(basis) == k:
        return basis
    for j in range(start, len(free)):
        v = free[j]
        if v in span:
            continue
        new = span | {add(v, s) for s in span}
        if new <= allowed:
            found = search(new, basis + [v], j + 1)
            if found:
                return found
    return None


if len(free) < 2 ** k - 1:
    print(f"fewer than {2**k - 1} free translations: no linear codeword can be added")
else:
    basis = search({zero}, [], 0)
    if basis is None:
        print(f"no {k}-dimensional subspace lies inside the free translations: "
              f"no linear codeword can be added to the clique")
    else:
        print(f"FOUND a {k}-dimensional subspace inside the free translations, "
              f"with basis {[''.join(map(str, v)) for v in basis]}: "
              f"the linear codeword on it can be added to the clique")
