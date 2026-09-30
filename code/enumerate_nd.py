#!/usr/bin/env python3
"""
enumerate_nd.py -- Exhaustive enumeration of the nondegenerate subgroups of
U_n of order 2^k, level by level.

Every group K of order 2^(j+1) has a subgroup H of index 2, and a subgroup of
index 2 is normal. Hence K = H u gH for some g outside H with g^2 in H and
g H g^-1 = H; conversely every such g yields a subgroup of order 2^(j+1).
Subgroups of a nondegenerate subgroup are nondegenerate, so the nondegenerate
subgroups of order 2^(j+1) are obtained, all of them, by extending those of
order 2^j. Starting from the trivial group, the script climbs one level at a
time up to order 2^k.

For each H the candidates g are read off a precomputed table of square roots
(the elements whose square lies in H), and only those are tested for
normalizing H. Elements of U_n are encoded as integers, with the product
computed from composition and action tables that are built from psc_gen, so
the operation is exactly that of the paper (eq:Un) and is checked against it.

    python enumerate_nd.py --n 6 --k 3
    python enumerate_nd.py --n 6 --k 3 --compare pool_663.pkl

Writes pool_<n><2k><k>_full.pkl, in the format of the other pools: a list of
codewords, each a list of [vector, permutation] pairs. The distance parameter
in the name is the maximum one, 2k; the pool itself does not depend on d.
"""
import argparse
import pickle
import random
import time
from itertools import permutations

import psc_gen

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, required=True)
ap.add_argument("--k", type=int, required=True)
ap.add_argument("--compare", help="a sampled pool (.pkl) to check against the "
                                  "complete enumeration")
ap.add_argument("--out", help="output file (default pool_<n><2k><k>_full.pkl)")
a = ap.parse_args()
n, k = a.n, a.k

t0 = time.time()
G = psc_gen.make(n)
perms = list(permutations(range(n)))
P = len(perms)
pidx = {p: i for i, p in enumerate(perms)}
V = 1 << n


def vec(x):
    return tuple((x >> (n - 1 - i)) & 1 for i in range(n))


vint = {vec(x): x for x in range(V)}
zero, idp = vec(0), tuple(range(n))
# comp[s][t] = index of s*t ; act[s][y] = s(y), both taken from psc_gen
comp = [[pidx[G['prod']((zero, perms[s]), (zero, perms[t]))[1]] for t in range(P)]
        for s in range(P)]
act = [[vint[G['prod']((zero, perms[s]), (vec(y), idp))[0]] for y in range(V)]
       for s in range(P)]
N = P * V
E = pidx[idp] * V            # element (x, s) is encoded as s*V + x


def mul(g, h):
    s, x = divmod(g, V)
    t, y = divmod(h, V)
    return comp[s][t] * V + (x ^ act[s][y])


def decode(g):
    s, x = divmod(g, V)
    return (vec(x), perms[s])


# the encoded product must agree with psc_gen on random pairs
rng = random.Random(0)
for _ in range(2000):
    g, h = rng.randrange(N), rng.randrange(N)
    assert decode(mul(g, h)) == G['prod'](decode(g), decode(h)), "encoding mismatch"

inv = [0] * N
for g in range(N):
    s, x = divmod(g, V)
    # (x,s)^-1 = (s^-1(x), s^-1); find it through the table
    for h in (pidx[G['perm_inv'](perms[s])] * V + y for y in range(V)):
        if mul(g, h) == E:
            inv[g] = h
            break
square = [mul(g, g) for g in range(N)]
roots = {}
for g in range(N):
    roots.setdefault(square[g], []).append(g)
print(f"U_{n}: {N:,} elements; tables built and checked ({time.time()-t0:.0f}s)")


def nondegenerate(K):
    return len({g % V for g in K}) == len(K)


level = {frozenset([E])}
for j in range(1, k + 1):
    t1 = time.time()
    nxt = set()
    for H in level:
        Hl = list(H)
        gens = [h for h in Hl if h != E]
        cands = set()
        for h in Hl:
            cands.update(roots.get(h, ()))
        for g in cands:
            if g in H:
                continue
            gi = inv[g]
            if all(mul(mul(g, h), gi) in H for h in gens):
                K = frozenset(Hl + [mul(g, h) for h in Hl])
                if nondegenerate(K):
                    nxt.add(K)
    level = nxt
    print(f"  order {2**j:>3}: {len(level):,} nondegenerate subgroups "
          f"({time.time()-t1:.0f}s)")

pool = [[[list(v), list(p)] for v, p in map(decode, sorted(K))] for K in level]
out = a.out or f"pool_{n}{2*k}{k}_full.pkl"
with open(out, "wb") as fh:
    pickle.dump(pool, fh)
print(f"written {out} ({len(pool):,} subgroups, {time.time()-t0:.0f}s in total)")

if a.compare:
    with open(a.compare, "rb") as fh:
        sampled = pickle.load(fh)
    full = {frozenset((tuple(v), tuple(p)) for v, p in H) for H in pool}
    samp = {frozenset((tuple(v), tuple(p)) for v, p in H) for H in sampled}
    print(f"\ncomparison with {a.compare}: {len(samp):,} sampled, "
          f"{len(samp & full):,} of them in the complete list, "
          f"{len(samp - full):,} NOT in it")
    print(f"  the sampled pool covers {100*len(samp)/len(full):.1f}% "
          f"of the {len(full):,} candidates")
    if samp - full:
        print("  *** a sampled candidate is missing from the enumeration: "
              "the enumeration is wrong")
