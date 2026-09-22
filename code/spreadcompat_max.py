#!/usr/bin/env python3
"""
spreadcompat_max.py -- Maximum number of pairwise disjoint spread-compatible codewords in a pool.

A nondegenerate codeword C <= U_n is SPREAD-COMPATIBLE when pi_v != id
for every nonzero v in supp(C), i.e. the identity permutation occurs only at
the zero vector. These are exactly the codewords compatible with the whole
Desarguesian spread, and Proposition (prodspread) lifts m pairwise disjoint
spread-compatible codewords of order 2^k to

    A^P_nd(2n, 4k, 2k)  >=  (2^{2n}-1)/(2^{2k}-1) + m,

so this count feeds the product construction directly. Raising m raises the
lifted bound verbatim.

Usage:
    python spreadcompat_max.py --pool pool_884.pkl --n 8 --k 4
    python spreadcompat_max.py --pool pool_663.pkl --n 6 --k 3 --time-limit 3600
"""
import argparse, json, os, pickle, random, time
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--pool", required=True)
ap.add_argument("--n", type=int, required=True)
ap.add_argument("--k", type=int, required=True)
ap.add_argument("--time-limit", type=int, default=3600)
ap.add_argument("--greedy-reps", type=int, default=3000)
ap.add_argument("--max-ilp", type=int, default=60000,
                help="if there are more spread-compatible candidates than this, the ILP "
                     "is restricted to the greedy seed and the candidates that clash "
                     "least with it")
ap.add_argument("--out", default=None)
a = ap.parse_args()

n, k = a.n, a.k
ORDER = 2 ** k
identity_perm = tuple(range(n))
zero = tuple([0] * n)
e = (zero, identity_perm)

def load(path):
    if path.endswith(".pkl"):
        data = pickle.load(open(path, "rb"))
    else:
        data = json.load(open(path))
        if isinstance(data, dict):
            for f in ("codes", "candidates", "pool"):
                if f in data: data = data[f]; break
    return [frozenset((tuple(v), tuple(p)) for v, p in H) for H in data]

t0 = time.time()
pool = load(a.pool)
pool = [H for H in pool if len(H) == ORDER and len({v for v, p in H}) == ORDER]
print(f"pool: {len(pool)} nondegenerate candidates of order {ORDER} "
      f"({time.time()-t0:.0f}s)")

# Spread-compatible filter: the identity permutation may occur only at the zero vector
spreadcompat = [H for H in pool if all(p != identity_perm for v, p in H if v != zero)]
print(f"Spread-compatible candidates: {len(spreadcompat)}  ({100*len(spreadcompat)/max(1,len(pool)):.1f}% of the pool)")
if not spreadcompat:
    print("no spread-compatible candidates; nothing to do"); raise SystemExit(0)

# The greedy pass runs on ALL SPREAD-COMPATIBLE candidates; the ILP is then restricted to
# the greedy solution plus the candidates most compatible with it. Handing CBC
# the whole spread-compatible set does not work: with tens of thousands of variables it does
# not finish building the model within any reasonable time limit and returns
# no feasible solution at all.
work = spreadcompat
cores = [H - {e} for H in work]
N = len(work)

t0 = time.time(); best = []
for s in range(a.greedy_reps):
    random.seed(s)
    order = list(range(N)); random.shuffle(order)
    used = set(); chosen = []
    for i in order:
        if cores[i] & used: continue
        chosen.append(i); used |= cores[i]
    if len(chosen) > len(best): best = chosen
print(f"greedy over all {N} spread-compatible candidates: {len(best)} disjoint ({time.time()-t0:.0f}s)")

# restrict the ILP to the seed plus its most compatible candidates
if N > a.max_ilp:
    t0 = time.time()
    seed_elements = set()
    for i in best: seed_elements |= cores[i]
    in_seed = set(best)
    scored = []
    for i in range(N):
        if i in in_seed: continue
        clashes = len(cores[i] & seed_elements)
        scored.append((clashes, i))
    scored.sort()
    room = max(0, a.max_ilp - len(best))
    keep = list(best) + [i for _, i in scored[:room]]
    print(f"  ILP restricted to {len(keep)} of {N} candidates "
          f"({time.time()-t0:.0f}s)")
else:
    keep = list(range(N))

remap = {orig: j for j, orig in enumerate(keep)}
best_local = [remap[i] for i in best]
cores_ilp = [cores[i] for i in keep]

import pulp
element_to = defaultdict(list)
for j, c in enumerate(cores_ilp):
    for el in c: element_to[el].append(j)
prob = pulp.LpProblem("spreadcompat", pulp.LpMaximize)
x = [pulp.LpVariable(f"x{j}", cat="Binary") for j in range(len(keep))]
prob += pulp.lpSum(x)
nres = 0
for owners in element_to.values():
    if len(owners) >= 2:
        prob += pulp.lpSum(x[j] for j in owners) <= 1; nres += 1
for j in best_local: x[j].setInitialValue(1)
print(f"ILP: {len(keep)} variables, {nres} constraints, limit {a.time_limit}s")
t0 = time.time()
prob.solve(pulp.PULP_CBC_CMD(msg=1, timeLimit=a.time_limit, warmStart=True))
status = pulp.LpStatus[prob.status]
# PuLP reports "Optimal" also when CBC stops on the time limit with a
# feasible solution; only sol_status tells a proven optimum apart.
if status == "Optimal" and prob.sol_status != pulp.LpSolutionOptimal:
    status = "Feasible (time limit, not proven optimal)"
sel = [keep[j] for j in range(len(keep)) if pulp.value(x[j]) and pulp.value(x[j]) > 0.5]
if len(sel) < len(best): sel = best; status += " (greedy seed kept)"
m = len(sel)
print(f"\n===== m = {m} pairwise disjoint spread-compatible codewords  [CBC {status}]  "
      f"({time.time()-t0:.0f}s) =====")

# verify
chosen = [work[i] for i in sel]
cc = [H - {e} for H in chosen]
ok = all(cc[i].isdisjoint(cc[j]) for i in range(m) for j in range(i+1, m))
allspreadcompat = all(all(p != identity_perm for v, p in H if v != zero) for H in chosen)

# m feeds a theorem through Proposition (prodspread), so the chosen codewords
# are re-verified here from first principles rather than trusted to the pool:
# closure under the twisted product, order 2^k and nondegeneracy.
def _perm_inv(p):
    s_ = [0]*n
    for i in range(n): s_[p[i]] = i
    return tuple(s_)

def _prod(x, y):
    (xv, xp), (yv, yp) = x, y
    q = _perm_inv(xp)
    pv = tuple(yv[q[i]] for i in range(n))
    return (tuple((xv[i]+pv[i]) % 2 for i in range(n)),
            tuple(xp[yp[i]] for i in range(n)))

ok_group = all(len(H) == ORDER and len({v for v, p in H}) == ORDER
               and all(_prod(u, w) in H for u in H for w in H) for H in chosen)
print(f"  pairwise disjoint: {ok} | all spread-compatible: {allspreadcompat} "
      f"| closed subgroups of order {ORDER}, nondegenerate: {ok_group}")
if not (ok and allspreadcompat and ok_group):
    raise SystemExit("FAILED verification: the value of m must not be used")

if n % k == 0:
    lifted = (2**(2*n) - 1)//(2**(2*k) - 1) + m
    print(f"\n  lift to ({2*n},{4*k},{2*k}):  A^P >= (2^{2*n}-1)/(2^{2*k}-1) + {m} "
          f"= {(2**(2*n)-1)//(2**(2*k)-1)} + {m} = {lifted}")
else:
    print(f"\n  no lift: the product construction requires k | n, and {k} does not divide {n}")

out = a.out or f"spreadcompat_{n}{2*k}{k}.json"
if os.path.exists(out):
    print(f"  NOTE: {out} already exists and will be overwritten")
json.dump([[[list(v), list(p)] for v, p in H] for H in chosen], open(out, "w"))
print(f"  written {out}")
