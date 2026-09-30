#!/usr/bin/env python3
"""
filter_cyclic.py -- Restrict a pool to its cyclic codewords, or report how many
codewords of a clique are cyclic.

A codeword of order 2^k is a 2-group, so every element has order dividing 2^k,
and the codeword is cyclic exactly when some element g satisfies
g^(2^(k-1)) != e. The test therefore costs k-1 squarings per element.

Restricting a COMPLETE pool (for instance one written by exact_k2.py) to its
cyclic members gives the complete pool of cyclic candidates, over which
exact_highs.py can then determine the cyclic-only optimum exactly: the linear
relaxation certifies an upper bound and the ILP supplies a matching clique.

    python filter_cyclic.py --pool pool_742.pkl --n 7 --k 2
    python filter_cyclic.py --clique clique_742_cyclic.json --n 7 --k 2
"""
import argparse
import json
import pickle
import time

import psc_gen

ap = argparse.ArgumentParser()
ap.add_argument("--pool", help="complete pool (.pkl) to restrict to cyclic codewords")
ap.add_argument("--clique", help="clique (.json or .pkl) whose cyclic codewords to count")
ap.add_argument("--n", type=int, required=True)
ap.add_argument("--k", type=int, required=True)
ap.add_argument("--out", help="output pool (default: <pool stem>_cyclic.pkl)")
a = ap.parse_args()
if bool(a.pool) == bool(a.clique):
    raise SystemExit("give exactly one of --pool and --clique")

G = psc_gen.make(a.n)
prod, e = G['prod'], G['e']


def is_cyclic(H):
    for v, p in H:
        g = (tuple(v), tuple(p))
        for _ in range(a.k - 1):
            g = prod(g, g)
        if g != e:
            return True
    return False


def load(path):
    if path.endswith(".json"):
        with open(path) as fh:
            return json.load(fh)
    with open(path, "rb") as fh:
        return pickle.load(fh)


t0 = time.time()
data = load(a.pool or a.clique)
cyclic = [H for H in data if is_cyclic(H)]
elapsed = time.time() - t0

if a.clique:
    print(f"{a.clique}: {len(data)} codewords, {len(cyclic)} cyclic "
          f"({'all' if len(cyclic) == len(data) else 'not all'}) ({elapsed:.0f}s)")
else:
    out = a.out or a.pool.rsplit(".", 1)[0] + "_cyclic.pkl"
    with open(out, "wb") as fh:
        pickle.dump(cyclic, fh)
    print(f"{a.pool}: {len(data):,} candidates, {len(cyclic):,} cyclic ({elapsed:.0f}s)")
    print(f"  written to {out}")
