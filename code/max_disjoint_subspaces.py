#!/usr/bin/env python3
"""
max_disjoint_subspaces.py -- Exact maximum number of pairwise disjoint
subspace supports occurring in a clique.

The cliques of Theorem thm:main contain many supports that happen to be
k-dimensional subspaces of F_2^n (15, 123, 343 and 774 at the four parameter
sets). A natural question is how many of them are pairwise disjoint, i.e. how
much of a spread they could form: a full spread has (2^n-1)/(2^k-1) members.

A greedy pass gives a lower bound; this script closes the question with an
ILP, so the resulting number is the exact maximum over the supports present
in the clique (not over all subspaces of F_2^n).

Two subspaces of the same dimension are disjoint when they meet only in the
zero vector, so the constraint matrix is built through an element-to-subspace
index: every nonzero vector may be used by at most one selected subspace.

Usage (from the folder holding the clique files):
    python max_disjoint_subspaces.py --clique clique_442.json   --n 4  --k 2
    python max_disjoint_subspaces.py --clique clique_663.json   --n 6  --k 3
    python max_disjoint_subspaces.py --clique clique_884.json   --n 8  --k 4
    python max_disjoint_subspaces.py --clique clique_10105.json --n 10 --k 5

Requires PuLP (pip install pulp).
"""
import argparse, json, random, time
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--clique", required=True)
ap.add_argument("--n", type=int, required=True)
ap.add_argument("--k", type=int, required=True)
ap.add_argument("--time-limit", type=int, default=600)
ap.add_argument("--greedy-reps", type=int, default=2000)
a = ap.parse_args()

n, k = a.n, a.k
zero = tuple([0] * n)

# ------------------------------------------------------------------ loading
data = json.load(open(a.clique))
if isinstance(data, dict):
    for f in ("codes", "candidates", "clique", "pool"):
        if f in data:
            data = data[f]
            break
clique = [[(tuple(v), tuple(p)) for v, p in C] for C in data]
print(f"clique: {len(clique)} codewords of order {2**k} in U_{n}")


def is_subspace(S):
    Sset = set(S)
    return all(tuple(x ^ y for x, y in zip(u, v)) in Sset for u in Sset for v in Sset)


t0 = time.time()
supports = {frozenset(v for v, p in C) for C in clique}
subspaces = sorted({s for s in supports if is_subspace(s)})
N = len(subspaces)
full_spread = (2 ** n - 1) // (2 ** k - 1)
print(f"distinct supports: {len(supports)} | subspace supports: {N} "
      f"({time.time()-t0:.0f}s)")
print(f"a full spread of F_2^{n} into {k}-subspaces has {full_spread} members\n")
if N == 0:
    raise SystemExit("no subspace supports; nothing to do")

cores = [s - {zero} for s in subspaces]

# ------------------------------------------------------------------ greedy
t0 = time.time()
best = []
for seed in range(a.greedy_reps):
    random.seed(seed)
    order = list(range(N))
    random.shuffle(order)
    used, chosen = set(), []
    for i in order:
        if cores[i] & used:
            continue
        chosen.append(i)
        used |= cores[i]
    if len(chosen) > len(best):
        best = chosen
print(f"greedy over {a.greedy_reps} restarts: {len(best)} disjoint "
      f"({time.time()-t0:.0f}s)")

# --------------------------------------------------------------------- ILP
import pulp

t0 = time.time()
elem_to = defaultdict(list)
for i, c in enumerate(cores):
    for el in c:
        elem_to[el].append(i)

prob = pulp.LpProblem("max_disjoint", pulp.LpMaximize)
x = [pulp.LpVariable(f"x{i}", cat="Binary") for i in range(N)]
prob += pulp.lpSum(x)
ncons = 0
for owners in elem_to.values():
    if len(owners) >= 2:
        prob += pulp.lpSum(x[i] for i in owners) <= 1
        ncons += 1
for i in best:
    x[i].setInitialValue(1)
print(f"ILP: {N} variables, {ncons} constraints, limit {a.time_limit}s")
prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=a.time_limit, warmStart=True))
status = pulp.LpStatus[prob.status]
sel = [i for i in range(N) if pulp.value(x[i]) and pulp.value(x[i]) > 0.5]
if len(sel) < len(best):
    sel, status = best, status + " (greedy kept)"
m = len(sel)

# ------------------------------------------------------------- verification
chosen = [subspaces[i] for i in sel]
ok = all(chosen[p] & chosen[q] == {zero}
         for p in range(m) for q in range(p + 1, m))
exact = (status == "Optimal")

print(f"\n===== maximum pairwise disjoint subspace supports: {m} "
      f"[CBC {status}] ({time.time()-t0:.0f}s) =====")
print(f"  pairwise disjoint verified: {ok}")
print(f"  {'EXACT over the supports in this clique' if exact else 'lower bound only'}")
print(f"  a full spread would have {full_spread}; "
      f"this is {100*m/full_spread:.0f}% of one")
