#!/usr/bin/env python3
"""
ilp_subpool.py -- Maximum-clique ILP over a RESTRICTED subpool.
===============================================================
Needed whenever the pool is large: handing CBC hundreds of thousands of
variables does not work. PuLP builds the model in Python, writes a huge .lp
file and CBC parses it, all of this BEFORE timeLimit starts counting (the
limit governs only the branch-and-bound phase). The process then looks hung
for hours.

The remedy is to solve the ILP only over

    (seed clique) + (candidates compatible with >= min_compat of the seed)

which are the only ones that can extend it. The optimum of the subpool is
always >= |seed|, so the bound never gets worse and the step can be iterated
with a growing subpool.

Unlike a fixed d=2k formulation, this accepts SUB-MAXIMUM distances: the
threshold |core_i cap core_j| <= 2^{(2k-d)/2} - 1 is computed from (n,d,k).

    python ilp_subpool.py --pool pool_763.pkl --n 7 --d 6 --k 3 --aq 17
    python ilp_subpool.py --pool pool_643.pkl --n 6 --d 4 --k 3 --aq 77 \
        --seed clique_643.pkl --min-compat 3100 --max-subpool 8000
"""
import argparse, json, pickle, os, random, time
from collections import defaultdict, Counter
import psc_gen

ap = argparse.ArgumentParser()
ap.add_argument("--pool", required=True)
ap.add_argument("--n", type=int, required=True)
ap.add_argument("--d", type=int, required=True)
ap.add_argument("--k", type=int, required=True)
ap.add_argument("--aq", type=int, required=True)
ap.add_argument("--seed", default=None, help="seed clique (.pkl/.json); greedy if omitted")
ap.add_argument("--greedy-reps", type=int, default=4000)
ap.add_argument("--min-compat", type=int, default=None,
                help="a candidate enters if compatible with >= this many of the seed "
                     "(default |seed|-25)")
ap.add_argument("--max-subpool", type=int, default=6000)
ap.add_argument("--time-limit", type=int, default=3600)
ap.add_argument("--reference", type=int, default=None,
                help="a reference value to print alongside the result (optional)")
ap.add_argument("--out-tag", default=None)
a = ap.parse_args()

n, d, k = a.n, a.d, a.k
ORDER = 2**k
max_shared = 2**((2*k - d)//2) - 1
G = psc_gen.make(n)
prod, e = G['prod'], G['e']
print(f"Case ({n},{d},{k}): order {ORDER}, threshold |core cap core| <= {max_shared}")

def load(path):
    if path.endswith(".pkl"):
        data = pickle.load(open(path, "rb"))
    else:
        data = json.load(open(path))
        if isinstance(data, dict):
            for f in ("candidates", "codes", "pool"):
                if f in data: data = data[f]; break
    return [frozenset((tuple(v), tuple(p)) for v, p in H) for H in data]

t0 = time.time()
pool = load(a.pool)
print(f"  pool: {len(pool)} candidates ({time.time()-t0:.0f}s)")
pool = [H for H in pool if len(H) == ORDER and G['nondeg'](H)]
core = [H - {e} for H in pool]
N = len(pool)
index_of = {H: i for i, H in enumerate(pool)}

# ---------- seed ----------
if a.seed and not os.path.exists(a.seed):
    raise SystemExit(f"ABORTED: seed file {a.seed} does not exist")
if a.seed:
    seed_words = load(a.seed)
    # Seed codewords absent from the pool are added: they are valid subgroups
    # and must be allowed into the final clique.
    missing = 0
    for H in seed_words:
        if H not in index_of and len(H) == ORDER and G['nondeg'](H):
            index_of[H] = len(pool); pool.append(H); core.append(H - {e}); missing += 1
    N = len(pool)
    seed_idx = [index_of[H] for H in seed_words if H in index_of]
    print(f"  seed: {len(seed_idx)} codewords"
          + (f" ({missing} were not in the pool and have been added)" if missing else ""))
else:
    t0 = time.time(); best = []
    for s in range(a.greedy_reps):
        random.seed(s); order = list(range(N)); random.shuffle(order)
        if max_shared == 0:
            used = set(); chosen = []
            for i in order:
                if core[i] & used: continue
                chosen.append(i); used |= core[i]
        else:
            # element -> already chosen positions, to avoid comparing
            # the candidate against the whole clique at each step
            occupied = defaultdict(list); chosen = []
            for i in order:
                shares = Counter()
                for el in core[i]:
                    for pos in occupied.get(el, ()): shares[pos] += 1
                if shares and max(shares.values()) > max_shared: continue
                pos = len(chosen); chosen.append(i)
                for el in core[i]: occupied[el].append(pos)
        if len(chosen) > len(best): best = chosen
    seed_idx = best
    print(f"  greedy seed: {len(seed_idx)} ({time.time()-t0:.0f}s)")

S = len(seed_idx)
min_compat = a.min_compat if a.min_compat is not None else max(1, S - 25)
print(f"  min-compat = {min_compat} (of {S})")

# ---------- subpool: compatibility against the seed, via an element index ----------
t0 = time.time()
elem_to_seed = defaultdict(list)
for pos, i in enumerate(seed_idx):
    for el in core[i]: elem_to_seed[el].append(pos)
in_seed = set(seed_idx)
scored = []
for i in range(N):
    if i in in_seed: continue
    shares = Counter()
    for el in core[i]:
        for pos in elem_to_seed.get(el, ()): shares[pos] += 1
    conflicts = sum(1 for c in shares.values() if c > max_shared)
    compat = S - conflicts
    if compat >= min_compat: scored.append((compat, i))
scored.sort(key=lambda t: -t[0])
room = max(0, a.max_subpool - S)
sub = list(seed_idx) + [i for _, i in scored[:room]]
print(f"  candidates passing the filter: {len(scored)} "
      f"(added {len(sub)-S}, cap {a.max_subpool})")
print(f"  subpool: {len(sub)} vertices ({time.time()-t0:.0f}s)")

# ---------- ILP ----------
import pulp
t0 = time.time()
prob = pulp.LpProblem("clique", pulp.LpMaximize)
x = [pulp.LpVariable(f"x{j}", cat="Binary") for j in range(len(sub))]
prob += pulp.lpSum(x)
n_constraints = 0
if max_shared == 0:
    elem_to = defaultdict(list)
    for j, i in enumerate(sub):
        for el in core[i]: elem_to[el].append(j)
    for owners in elem_to.values():
        if len(owners) >= 2: prob += pulp.lpSum(x[j] for j in owners) <= 1; n_constraints += 1
else:
    for p in range(len(sub)):
        for q in range(p+1, len(sub)):
            if len(core[sub[p]] & core[sub[q]]) > max_shared:
                prob += x[p] + x[q] <= 1; n_constraints += 1
for j in range(S): x[j].setInitialValue(1)
print(f"  {n_constraints} constraints ({time.time()-t0:.0f}s), solving (limit {a.time_limit}s)...")
t0 = time.time()
try:
    prob.solve(pulp.PULP_CBC_CMD(msg=1, timeLimit=a.time_limit, warmStart=True))
    status = pulp.LpStatus[prob.status]
    # PuLP reports "Optimal" also when CBC stops on the time limit with a
    # feasible solution; only sol_status tells a proven optimum apart.
    if status == "Optimal" and prob.sol_status != pulp.LpSolutionOptimal:
        status = "Feasible (time limit, not proven optimal)"
    selected = [sub[j] for j in range(len(sub)) if pulp.value(x[j]) and pulp.value(x[j]) > 0.5]
except Exception as ex:
    print(f"  CBC failed: {ex}"); selected = []; status = "CBC error"

def is_valid(s):
    return all(len(core[s[p]] & core[s[q]]) <= max_shared
               for p in range(len(s)) for q in range(p+1, len(s)))
if len(selected) > S and is_valid(selected):
    chosen_set, method = selected, f"ILP subpool ({status})"
else:
    chosen_set, method = seed_idx, f"seed (ILP {status} did not improve)"
value = len(chosen_set)
print(f"\n===== A^P({n},{d},{k}) = {value}  [{method}]  ({time.time()-t0:.0f}s) =====")

clique = [pool[i] for i in chosen_set]
cores_sel = [H - {e} for H in clique]
ok_dist = all(len(cores_sel[p] & cores_sel[q]) <= max_shared for p in range(value) for q in range(p+1, value))
ok_group = all(all(prod(u, w) in H for u in H for w in H) for H in clique)
n_linear = sum(1 for H in clique if G['is_linear'](H))
print(f"  verification: distances={ok_dist} groups={ok_group} linear={n_linear}")
if not (ok_dist and ok_group):
    raise SystemExit("ABORTED: the clique failed verification; nothing written")
print(f"  factor {value/a.aq:.1f}x  (A_q={a.aq}" + (f", reference value {a.reference}" if a.reference else "") + ")")

tag = a.out_tag or f"{n}{d}{k}"
out = f"clique_{tag}.json"
if os.path.exists(out):
    print(f"  NOTE: {out} already exists and will be overwritten")
serialized = [[[list(v), list(p)] for v, p in sorted(H)] for H in clique]
pickle.dump(serialized, open(f"clique_{tag}.pkl", "wb"))
json.dump(serialized, open(out, "w"))
json.dump({"case": f"({n},{d},{k})", "A_P": value, "method": method,
           "operation": "as in the paper, eq:Un", "pool_size": N,
           "subpool_size": len(sub), "min_compat": min_compat,
           "verification_distances": ok_dist, "verification_groups": ok_group,
           "linear": n_linear, "nonlinear": value-n_linear,
           "reference_value": a.reference, "A_q": a.aq},
          open(f"results_{tag}.json", "w"), indent=2, ensure_ascii=False)
print(f"  written: results_{tag}.json, clique_{tag}.{{pkl,json}}")
