#!/usr/bin/env python3
"""
exact_highs.py -- Maximum-clique ILP over a COMPLETE pool, solved in memory
with HiGHS (via scipy.optimize.milp) instead of CBC through PuLP.

Why: PuLP hands the model to CBC by writing an MPS file to disk. At (8,4,2)
the exhaustive pool has 4.8 million candidates and that file is around
500 MB; CBC aborts while reading it. HiGHS receives the constraint matrix as
a sparse array in memory, so no file is written at all.

The pool is the one written by exact_k2.py (or any other complete pool), so
the expensive enumeration does not have to be repeated.

Two modes:

  default   solve the ILP. If HiGHS proves optimality the value is EXACT;
            if it stops on the time limit it reports the interval
            [best feasible, dual bound].

  --lp-only solve the linear relaxation only. Over an EXHAUSTIVE pool its
            optimum is a valid UPPER bound on A^P(n,d,k), which is what the
            chain of bounds needs. Much cheaper than the ILP.

            The relaxation is solved by the interior-point method with
            crossover switched off. These set-packing LPs are highly
            degenerate, and the dual simplex tends to stall for a long time
            with its objective already fixed while it clears primal
            infeasibility; the interior-point method does not suffer from this.

            The reported bound does not rely on the solver's tolerances. From
            the row duals y returned by HiGHS the script builds an explicitly
            dual-feasible pair (y, z), with y >= 0 and z = max(0, 1 - A^T y),
            and reports sum(y) + sum(z). By weak duality this is an upper bound
            on the LP, hence on A^P, whatever the accuracy of y.

    python exact_highs.py --pool pool_842.pkl --n 8 --d 4 --k 2 --aq 85 --lp-only
    python exact_highs.py --pool pool_842.pkl --n 8 --d 4 --k 2 --aq 85 --time-limit 86400

Only the maximum-distance case d = 2k is supported: there distinct codewords
meet only in the identity, so one constraint per shared element suffices.
"""
import argparse
import json
import pickle
import time
from collections import defaultdict

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix

ap = argparse.ArgumentParser()
ap.add_argument("--pool", required=True)
ap.add_argument("--n", type=int, required=True)
ap.add_argument("--d", type=int, required=True)
ap.add_argument("--k", type=int, required=True)
ap.add_argument("--aq", type=int, required=True)
ap.add_argument("--time-limit", type=float, default=3600)
ap.add_argument("--lp-only", action="store_true",
                help="solve only the linear relaxation (upper bound)")
ap.add_argument("--out-tag", default=None)
a = ap.parse_args()

n, d, k = a.n, a.d, a.k
if d != 2 * k:
    raise SystemExit("only d = 2k is supported (codewords meeting only in e)")
ORDER = 2 ** k
e = (tuple([0] * n), tuple(range(n)))

# --------------------------------------------------------------- load pool
t0 = time.time()
with open(a.pool, "rb") as fh:
    raw = pickle.load(fh)
pool = [frozenset((tuple(v), tuple(p)) for v, p in H) for H in raw]
del raw
N = len(pool)
print(f"pool: {N:,} candidates ({time.time()-t0:.0f}s)")

# --------------------------------------------------- sparse constraint matrix
# One row per element of U_n shared by at least two candidates; a row with a
# single candidate would only restate x <= 1, which the bounds already say.
t0 = time.time()
owners = defaultdict(list)
for j, H in enumerate(pool):
    for el in H:
        if el != e:
            owners[el].append(j)
rows, cols = [], []
r = 0
for js in owners.values():
    if len(js) >= 2:
        rows.extend([r] * len(js))
        cols.extend(js)
        r += 1
del owners
A = csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(r, N))
del rows, cols
print(f"constraints: {r:,} rows, {A.nnz:,} nonzeros ({time.time()-t0:.0f}s)")

# ------------------------------------------------------------------- solve
c = -np.ones(N)                                # maximize sum x
cons = LinearConstraint(A, -np.inf, 1)
integ = np.zeros(N) if a.lp_only else np.ones(N)
opts = {"time_limit": a.time_limit, "disp": True}

if a.lp_only:
    import highspy
    print(f"\nsolving the linear relaxation with HiGHS, interior point, "
          f"no crossover (limit {a.time_limit:.0f}s)...")
    Acsc = A.tocsc()
    lp = highspy.HighsLp()
    lp.num_col_ = N
    lp.num_row_ = Acsc.shape[0]
    lp.col_cost_ = np.full(N, -1.0)                    # minimize -sum x
    lp.col_lower_ = np.zeros(N)
    lp.col_upper_ = np.ones(N)
    lp.row_lower_ = np.full(lp.num_row_, -highspy.kHighsInf)
    lp.row_upper_ = np.ones(lp.num_row_)
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = Acsc.indptr.astype(np.int32)
    lp.a_matrix_.index_ = Acsc.indices.astype(np.int32)
    lp.a_matrix_.value_ = Acsc.data.astype(np.float64)
    h = highspy.Highs()
    h.setOptionValue("solver", "ipm")
    h.setOptionValue("run_crossover", "off")
    h.setOptionValue("time_limit", float(a.time_limit))
    # Presolve is switched off: with crossover also off, HiGHS cannot always
    # postsolve an interior-point solution, and then returns duals that are
    # incomplete for the original model. The bound computed below would still
    # be valid, but needlessly weak.
    h.setOptionValue("presolve", "off")
    h.passModel(lp)
    t0 = time.time()
    h.run()
    elapsed = time.time() - t0
    info = h.getInfo()
    print(f"  HiGHS status: {h.modelStatusToString(h.getModelStatus())}  ({elapsed:.0f}s)")
    print(f"  solver objective: {-info.objective_function_value:.6f}")

    # Rigorous bound by weak duality. The dual of  max 1.x, Ax<=1, 0<=x<=1  is
    # min 1.y + 1.z, A^T y + z >= 1, y,z >= 0. Any y >= 0 completed by
    # z = max(0, 1 - A^T y) is dual feasible, so 1.y + 1.z bounds the LP.
    # Both sign conventions for the returned duals are tried; each gives a
    # valid bound, and the smaller is kept.
    row_dual = np.asarray(h.getSolution().row_dual)
    best = np.inf
    for y in (np.maximum(0.0, -row_dual), np.maximum(0.0, row_dual)):
        z = np.maximum(0.0, 1.0 - Acsc.T.dot(y))
        best = min(best, y.sum() + z.sum())
    ub = int(np.floor(best + 1e-9))
    print(f"\n===== certified LP bound: {best:.6f}  ({elapsed:.0f}s) =====")
    print(f"  UPPER bound: A^P({n},{d},{k}) <= {ub}")
    print(f"  (weak duality from an explicit dual-feasible point; valid only")
    print(f"   because the pool is exhaustive)")
    print(f"  ceiling on the factor: {ub/a.aq:.2f}x over A_q={a.aq}")
    raise SystemExit(0)

mode = "ILP"
print(f"\nsolving the {mode} with HiGHS (limit {a.time_limit:.0f}s)...")
t0 = time.time()
res = milp(c, constraints=cons, integrality=integ, bounds=Bounds(0, 1),
           options=opts)
elapsed = time.time() - t0

if res.x is None:
    raise SystemExit(f"no feasible solution: {res.message}")

sel = [j for j in range(N) if res.x[j] > 0.5]
value = len(sel)
dual = getattr(res, "mip_dual_bound", None)
proven = (res.status == 0)
upper = int(np.floor(-dual + 1e-6)) if dual is not None else None

print(f"\n===== A^P({n},{d},{k}) = {value}  ({elapsed:.0f}s) =====")
if proven:
    print("  EXACT VALUE (HiGHS proved optimality)")
else:
    print(f"  stopped: {res.message}")
    if upper is not None:
        print(f"  the optimum lies in [{value}, {upper}]")

# --------------------------------------------------------------- verification
clique = [pool[j] for j in sel]
cores = [H - {e} for H in clique]
seen = set()
disjoint = True
for c_ in cores:
    if c_ & seen:
        disjoint = False
        break
    seen |= c_
print(f"  pairwise disjoint: {disjoint}")
if not disjoint:
    raise SystemExit("ABORTED: the selection failed verification; nothing written")
print(f"  factor {value/a.aq:.2f}x  (A_q={a.aq})")

tag = a.out_tag or f"{n}{d}{k}_highs"
ser = [[[list(v), list(p)] for v, p in sorted(H)] for H in clique]
with open(f"clique_{tag}.pkl", "wb") as fh:
    pickle.dump(ser, fh)
with open(f"clique_{tag}.json", "w") as fh:
    json.dump(ser, fh)
with open(f"results_{tag}.json", "w") as fh:
    json.dump({"case": f"({n},{d},{k})", "A_P": value, "exact": proven,
               "upper_bound": upper, "solver": "HiGHS via scipy.optimize.milp",
               "pool_size": N, "A_q": a.aq}, fh, indent=2)
print(f"  written: results_{tag}.json, clique_{tag}.{{pkl,json}}")
