#!/usr/bin/env python3
"""
structural_stats.py -- Recompute every structural invariant reported in the
paper, over a given clique, and print the results as LaTeX-ready fragments.

The structural sections of the paper (support structure, isomorphism types,
rank/kernel pairs, conjugacy classes) are properties of the *particular*
clique they were computed on. When a clique is replaced by a larger one they
must all be recomputed; swapping the cardinality is not enough.

Usage:
    python structural_stats.py --clique clique_663.json  --n 6  --k 3
    python structural_stats.py --clique clique_884.json  --n 8  --k 4
    python structural_stats.py --clique clique_10105.json --n 10 --k 5
    python structural_stats.py --clique clique_442.json  --n 4  --k 2 --exact-classes
    python structural_stats.py --clique clique_663.json  --n 6  --k 3 --exact-classes

Add --exact-classes to count conjugacy classes exactly, by conjugating every
codeword by every element of U_n; this is how the class counts reported in the
paper were obtained, and it is feasible for n <= 6. For larger n, --classes
runs a randomized search instead, which gives an upper bound on the number of
classes together with a lower bound from isomorphism invariants.
"""
import argparse
import json
import math
import pickle
import random
from collections import Counter

ap = argparse.ArgumentParser()
ap.add_argument("--clique", required=True)
ap.add_argument("--n", type=int, required=True)
ap.add_argument("--k", type=int, required=True)
ap.add_argument("--classes", action="store_true",
                help="also estimate conjugacy classes (slow)")
ap.add_argument("--cross", action="store_true",
                help="cross-tabulate d(G) against exponent, as a LaTeX table")
ap.add_argument("--exact-classes", action="store_true",
                help="conjugate by EVERY element of U_n: exact class count, "
                     "feasible for n <= 6 (|U_4|=384, |U_6|=46080)")
ap.add_argument("--class-samples", type=int, default=2000)
a = ap.parse_args()

n, k = a.n, a.k
ORDER = 2 ** k
idp = tuple(range(n))
zero = tuple([0] * n)
e = (zero, idp)


# ----------------------------------------------------------------- group ops
_INV_CACHE = {}


def perm_inv(p):
    q = _INV_CACHE.get(p)
    if q is None:
        s = [0] * n
        for i in range(n):
            s[p[i]] = i
        q = tuple(s)
        _INV_CACHE[p] = q
    return q


def perm_act(p, v):
    # perm_inv is memoized: the exhaustive conjugacy search calls this ~10^8
    # times over a few thousand distinct permutations.
    q = perm_inv(p)
    return tuple(v[q[i]] for i in range(n))


def perm_mul(p, q):
    return tuple(p[q[i]] for i in range(n))


def prod(x, y):
    (a1, p), (b1, q) = x, y
    pv = perm_act(p, b1)
    return (tuple((a1[i] + pv[i]) % 2 for i in range(n)), perm_mul(p, q))


def order_of(g):
    x = g
    o = 1
    while x != e:
        x = prod(x, g)
        o += 1
        if o > 2 * ORDER:
            return -1
    return o


# ----------------------------------------------------------------- loading
def load(path):
    if path.endswith(".pkl"):
        data = pickle.load(open(path, "rb"))
    else:
        data = json.load(open(path))
        if isinstance(data, dict):
            for f in ("codes", "candidates", "clique", "pool"):
                if f in data:
                    data = data[f]
                    break
    return [frozenset((tuple(v), tuple(p)) for v, p in H) for H in data]


clique = load(a.clique)
M = len(clique)
if M == 0:
    raise SystemExit(f"ABORTED: {a.clique} contains no codewords")
print(f"clique: {M} codewords of order {ORDER} in U_{n}\n")


# ----------------------------------------------------- support structure
def support(H):
    return frozenset(v for v, p in H)


def is_subspace(S):
    S = set(S)
    return all(tuple((x[i] + y[i]) % 2 for i in range(n)) in S
               for x in S for y in S)


sups = [support(H) for H in clique]
by_support = Counter(sups)
distinct = len(by_support)
subspace_sups = sum(1 for s in by_support if is_subspace(s))
multi = Counter(c for c in by_support.values() if c >= 2)

print("=" * 62)
print("tab:support  --  support structure")
print("=" * 62)
print(f"  total M ................. {M}")
print(f"  distinct supports ....... {distinct}")
print(f"  subspace supports ....... {subspace_sups}")
print(f"  nonlinear supports ...... {distinct - subspace_sups}")
if multi:
    detail = ", ".join(f"{cnt} supports with {mult} structures"
                       for mult, cnt in sorted(multi.items()))
    print(f"  multi-structure ......... {detail}")
    print(f"  (max structures on one support: {max(by_support.values())})")
else:
    print("  multi-structure ......... 0")
mx = max(by_support.values())
cell = (f"${sum(multi.values())}$ supports with $\\ge 2$ structures"
        if len(multi) > 1 else
        (f"${multi[mx]}$ supports with ${mx}$ structures" if multi else "$0$"))
print("\n  LaTeX row:")
print(f"  $({n},{2*k},{k})$ & ${M}$ & ${distinct}$ & ${subspace_sups}$ & {cell} \\\\")


# ------------------------------------------------------- isomorphism types
def invariants(H):
    """Order profile, abelianness, |Z|, d(G) -- enough to name small groups."""
    S = set(H)
    orders = Counter(order_of(g) for g in S)
    abelian = all(prod(x, y) == prod(y, x) for x in S for y in S)
    Z = sum(1 for g in S if all(prod(g, h) == prod(h, g) for h in S))
    return orders, abelian, Z


def name_order8(H):
    orders, abelian, Z = invariants(H)
    if orders.get(8, 0) > 0:
        return "Z_8"
    if abelian:
        return "Z_2^3" if orders.get(2, 0) == 7 else "Z_4xZ_2"
    return "Q_8" if orders.get(2, 0) == 1 else "D_4"


def frattini_rank(H):
    """d(G) = log2 |G : Phi(G)| for a 2-group."""
    Hl = list(H)
    inv = {}
    for g in Hl:
        for h in Hl:
            if prod(g, h) == e:
                inv[g] = h
                break
    gen = {prod(prod(x, y), inv[prod(y, x)]) for x in Hl for y in Hl}
    gen |= {prod(g, g) for g in Hl}
    Phi = {e}
    frontier = list(gen)
    while frontier:
        x = frontier.pop()
        if x in Phi:
            continue
        Phi.add(x)
        for y in list(Phi):
            for pr in (prod(x, y), prod(y, x)):
                if pr not in Phi:
                    frontier.append(pr)
    return int(math.log2(len(H) // len(Phi)))


print("\n" + "=" * 62)
print("isomorphism types")
print("=" * 62)
if k == 3:
    types = Counter(name_order8(H) for H in clique)
    row = " & ".join(f"${types.get(t,0)}$"
                     for t in ("D_4", "Z_4xZ_2", "Z_8", "Q_8", "Z_2^3"))
    print(f"  D_4={types.get('D_4',0)}  Z_4xZ_2={types.get('Z_4xZ_2',0)}  "
          f"Z_8={types.get('Z_8',0)}  Q_8={types.get('Q_8',0)}  "
          f"Z_2^3={types.get('Z_2^3',0)}")
    print(f"\n  LaTeX row:  {row} \\\\")
    absent = [t for t in ("D_4", "Z_4xZ_2", "Z_8", "Q_8", "Z_2^3")
              if types.get(t, 0) == 0]
    print(f"  absent classes: {absent if absent else 'none'}")
elif k == 2:
    types = Counter("Z_4" if invariants(H)[0].get(4, 0) > 0 else "V_4"
                    for H in clique)
    print(f"  V_4={types.get('V_4',0)}  Z_4={types.get('Z_4',0)}")
else:
    ab = sum(1 for H in clique if invariants(H)[1])
    exps = Counter(max(invariants(H)[0]) for H in clique)
    dg = Counter(frattini_rank(H) for H in clique)
    ea = sum(1 for H in clique if all(order_of(g) <= 2 for g in H))
    print(f"  abelian ................. {ab} of {M} ({100*ab/M:.0f}%)")
    print(f"  nonabelian .............. {M-ab} ({100*(M-ab)/M:.0f}%)")
    print(f"  elementary abelian Z_2^{k} . {ea}")
    print(f"  exponent distribution ... {dict(sorted(exps.items()))}")
    print(f"  d(G) distribution ....... {dict(sorted(dg.items()))}")


# ------------------------------------------------- d(G) vs exponent table
if a.cross:
    print("\n" + "=" * 62)
    print("d(G) against exponent")
    print("=" * 62)
    cells = Counter()
    for H in clique:
        orders, abelian, Z = invariants(H)
        cells[(frattini_rank(H), max(orders))] += 1
    dvals = sorted({d for d, _ in cells})
    evals = sorted({e_ for _, e_ in cells})
    head = " ".join(f"{e_:>7}" for e_ in evals)
    print(f"  {'d(G)':>5} {head}  {'total':>7}")
    for d_ in dvals:
        row = " ".join(f"{cells.get((d_, e_), 0):>7}" for e_ in evals)
        tot = sum(cells.get((d_, e_), 0) for e_ in evals)
        print(f"  {d_:>5} {row}  {tot:>7}")
    tots = " ".join(f"{sum(cells.get((d_, e_),0) for d_ in dvals):>7}" for e_ in evals)
    print(f"  {'total':>5} {tots}  {M:>7}")

    print("\n  LaTeX:")
    print("  \\begin{tabular}{l" + "r"*len(evals) + "r}")
    print("  \\toprule")
    print("  $\\grank(C^*)$ & " + " & ".join(f"$\\exp={e_}$" for e_ in evals) + " & Total \\\\")
    print("  \\midrule")
    for d_ in dvals:
        row = " & ".join(f"${cells.get((d_, e_), 0)}$" for e_ in evals)
        tot = sum(cells.get((d_, e_), 0) for e_ in evals)
        print(f"  ${d_}$ & {row} & ${tot}$ \\\\")
    print("  \\midrule")
    print("  Total & " + " & ".join(f"${sum(cells.get((d_,e_),0) for d_ in dvals)}$" for e_ in evals) + f" & ${M}$ \\\\")
    print("  \\bottomrule")
    print("  \\end{tabular}")

# ------------------------------------------------------- rank and kernel
def rank_of(S):
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


def kernel_dim(S):
    Sset = set(S)
    ker = [v for v in Sset
           if all(tuple((v[i] + u[i]) % 2 for i in range(n)) in Sset
                  for u in Sset)]
    return int(math.log2(len(ker)))


print("\n" + "=" * 62)
print("tab:rk_dim  --  (rank, kernel dimension) pairs")
print("=" * 62)
lin_pairs, nonlin_pairs = Counter(), Counter()
for H in clique:
    S = support(H)
    pair = (rank_of(S), kernel_dim(S))
    (lin_pairs if is_subspace(S) else nonlin_pairs)[pair] += 1
fmt = lambda c: ", ".join(f"$({r},{d})\\times {m}$"
                          for (r, d), m in sorted(c.items(), key=lambda t: -t[1]))
print(f"  linear support ....... {fmt(lin_pairs) if lin_pairs else 'none'}")
print(f"  nonlinear support .... {fmt(nonlin_pairs) if nonlin_pairs else 'none'}")
print("\n  LaTeX row:")
print(f"  $({n},{2*k},{k})$, $M={M}$ & {fmt(lin_pairs)} & {fmt(nonlin_pairs)} \\\\")


# ------------------------------------------------------- conjugacy classes
if a.exact_classes:
    from itertools import permutations as _perms, product as _prod
    print("\n" + "=" * 62)
    print("conjugacy classes (EXHAUSTIVE -- exact)")
    print("=" * 62)
    allU = [(v, p) for v in _prod((0, 1), repeat=n) for p in _perms(range(n))]
    print(f"  conjugating by all {len(allU)} elements of U_{n} ...")
    index = {H: i for i, H in enumerate(clique)}
    parent = list(range(M))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj: parent[ri] = rj

    for i, H in enumerate(clique):
        for g in allU:
            gi = (perm_act(perm_inv(g[1]), g[0]), perm_inv(g[1]))
            K = frozenset(prod(prod(g, h), gi) for h in H)
            if K in index:
                union(i, index[K])
    sizes = Counter(find(i) for i in range(M))
    print(f"  EXACT: {len(sizes)} conjugacy classes among {M} codewords")
    print(f"  class sizes: {sorted(sizes.values(), reverse=True)}")
    print(f"  singletons: {sum(1 for s in sizes.values() if s == 1)}")

if a.classes:
    print("\n" + "=" * 62)
    print("conjugacy classes (randomized)")
    print("=" * 62)
    from itertools import permutations
    allp = list(permutations(range(n))) if n <= 8 else None
    rng = random.Random(0)

    def rand_elt():
        v = tuple(rng.randint(0, 1) for _ in range(n))
        if allp is not None:
            p = rng.choice(allp)
        else:
            pl = list(range(n)); rng.shuffle(pl); p = tuple(pl)
        return (v, p)

    def conj(H, g):
        gi = None
        x = g
        while True:
            nxt = prod(x, g)
            if nxt == e:
                gi = x; break
            x = nxt
        return frozenset(prod(prod(g, h), gi) for h in H)

    index = {H: i for i, H in enumerate(clique)}
    parent = list(range(M))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj: parent[ri] = rj

    for i, H in enumerate(clique):
        for _ in range(a.class_samples):
            g = rand_elt()
            K = conj(H, g)
            if K in index:
                union(i, index[K])
    classes = len({find(i) for i in range(M)})
    sizes = Counter(find(i) for i in range(M))
    top = sorted(sizes.values(), reverse=True)[:6]
    # The search merges only PROVEN conjugations, so a missed conjugation
    # leaves one class split in two: the partition obtained is finer than the
    # true one and the count is an UPPER bound on the number of classes.
    print(f"  UPPER bound: at most {classes} classes among {M} codewords")
    print("    (merges only proven conjugations, so missed ones inflate the count;")
    print("     raising --class-samples can only lower this number)")
    print(f"  largest class sizes: {top}")
    print(f"  singletons: {sum(1 for s in sizes.values() if s == 1)}")

    # A genuine LOWER bound: conjugation is an isomorphism, so codewords of
    # different isomorphism type are certainly in different classes.
    if k == 3:
        seen = {name_order8(H) for H in clique}
    elif k == 2:
        seen = {"Z_4" if invariants(H)[0].get(4, 0) > 0 else "V_4"
                for H in clique}
    else:
        seen = {(tuple(sorted(invariants(H)[0].items())), invariants(H)[1],
                 invariants(H)[2]) for H in clique}
    print(f"  LOWER bound: at least {len(seen)} classes "
          f"(distinct isomorphism invariants)")

print()
