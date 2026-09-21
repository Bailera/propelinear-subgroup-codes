#!/usr/bin/env python3
"""
build_lift.py -- Construct and verify the lifted families of the
product--spread construction (Proposition prodspread / Corollary n16).

Input: a file with m pairwise disjoint SPREAD-COMPATIBLE codewords of order
2^k in U_n (the output of spreadcompat_max.py). The script

  1. re-verifies the base codewords from first principles: subgroup closure,
     order 2^k, nondegeneracy, spread-compatibility, pairwise disjointness;
  2. builds the Desarguesian spread of F_2^{2n} into 2k-dimensional
     subspaces, embedded as linear codewords of U_{2n}
     (F_2^{2n} is viewed as F_q^{n/k} with q = 2^{2k}; the spread members
     are the F_q-lines through the origin);
  3. builds the diagonal lifts D_i = iota(C_i x C_i) <= U_{2n};
  4. verifies the WHOLE lifted family at parameters (2n, 4k, 2k): every
     codeword closed, of order 2^{2k}, nondegenerate; every pair meeting
     only in the identity; and every D_i spread-compatible in U_{2n}
     (which is what Corollary tower needs to iterate);
  5. writes clique_<tag>_lift.json in the standard format, so the family can
     be re-verified independently with verify_psc.py. The "_lift" suffix keeps
     the lifted families apart from the computed cliques: lifting the (4,4,2)
     base yields tag 884, which would otherwise collide with clique_884.json.

The group operation is recomputed here with the paper's convention
(eq:Un): (x,p)(y,q) = (x + p(y), p*q), p(v)_i = v_{p^{-1}(i)},
(p*q)(i) = p(q(i)).

Subgroup closure is certified by exhibiting a generating set: a subset
X of H is grown greedily until the Cayley closure <X> (BFS from e by
right multiplication) equals H; since <X> is a subgroup, equality proves
that H is one. This costs O(|H| * |X|) products instead of O(|H|^2).

Usage:
    python build_lift.py --base spreadcompat_663.json  --n 6  --k 3
    python build_lift.py --base spreadcompat_884.json  --n 8  --k 4
    python build_lift.py --base spreadcompat_10105.json --n 10 --k 5
"""
import argparse, json, os, pickle, sys, time

# ------------------------------------------------------------ group U_N
def make_group(N):
    idp = tuple(range(N))
    e = (tuple([0] * N), idp)

    def perm_inv(p):
        s = [0] * N
        for i in range(N):
            s[p[i]] = i
        return tuple(s)

    def perm_act(p, v):
        q = perm_inv(p)
        return tuple(v[q[i]] for i in range(N))

    def perm_mul(p, q):
        return tuple(p[q[i]] for i in range(N))

    def prod(a, b):
        (x, p), (y, q) = a, b
        py = perm_act(p, y)
        return (tuple((x[i] + py[i]) % 2 for i in range(N)), perm_mul(p, q))

    return e, idp, prod


def check_associativity(N, prod, trials=2000, seed=12345):
    import random
    rng = random.Random(seed)

    def rand():
        v = tuple(rng.randint(0, 1) for _ in range(N))
        pl = list(range(N)); rng.shuffle(pl)
        return (v, tuple(pl))

    for _ in range(trials):
        a, b, c = rand(), rand(), rand()
        if prod(prod(a, b), c) != prod(a, prod(b, c)):
            return False
    return True


def certify_subgroup(H, e, prod, order):
    """True iff H is a subgroup of the expected order, certified by a
    generating set (Cayley-graph BFS by right multiplication)."""
    if len(H) != order or e not in H:
        return False
    gens = []
    S = {e}
    for h in H:                       # greedy generating set
        if h in S:
            continue
        gens.append(h)
        frontier = list(S)
        while frontier:               # close S under right mult by gens
            x = frontier.pop()
            for g in gens:
                y = prod(x, g)
                if y not in S:
                    if len(S) >= order and y not in H:
                        return False
                    S.add(y)
                    if y not in H:
                        return False
                    frontier.append(y)
        if len(S) == order:
            break
    return S == set(H)


# ------------------------------------------------- GF(2^m) and the spread
IRREDUCIBLE = {2: 0b111, 4: 0b10011, 6: 0b1000011, 8: 0b100011101,
               10: 0b10000001001, 12: 0b1000001010011,
               14: 0b100010000100011, 16: 0b10001000000001011}


def pdeg(a):
    return a.bit_length() - 1


def pmod(a, f):
    df = pdeg(f)
    while a.bit_length() - 1 >= df:
        a ^= f << (a.bit_length() - 1 - df)
    return a


def pmul(a, b, f):
    r = 0
    while b:
        if b & 1:
            r ^= a
        b >>= 1
        a <<= 1
        a = pmod(a, f)
    return pmod(r, f)


def poly_gcd(a, b):
    while b:
        if a.bit_length() < b.bit_length():
            a, b = b, a
            continue
        a = pmod(a, b)
        a, b = b, a
    return a


def is_irreducible(f):
    """f irreducible over F_2 iff x^(2^m) = x mod f and, for every prime
    p | m, gcd(x^(2^(m/p)) - x, f) = 1."""
    m = pdeg(f)
    t = 0b10                          # the polynomial x
    for _ in range(m):
        t = pmul(t, t, f)
    if t != 0b10:
        return False
    primes = [p for p in range(2, m + 1) if m % p == 0 and
              all(p % q for q in range(2, p))]
    for p in primes:
        t = 0b10
        for _ in range(m // p):
            t = pmul(t, t, f)
        if poly_gcd(t ^ 0b10, f) != 1:
            return False
    return True


def desarguesian_spread(n2, k2):
    """The 2k-dimensional subspaces of a Desarguesian spread of F_2^{2n},
    as frozensets of {0,1}-tuples of length 2n. Here n2 = 2n, k2 = 2k and
    k2 | n2. F_2^{n2} is identified with F_q^{r}, q = 2^{k2}, r = n2/k2,
    blockwise (bit t of block j = coefficient of x^t); the spread members
    are the F_q-lines through the origin."""
    assert n2 % k2 == 0
    r = n2 // k2
    f = IRREDUCIBLE[k2]
    assert is_irreducible(f), f"polynomial {bin(f)} is not irreducible"
    q = 2 ** k2

    def to_bits(coords):
        v = []
        for c in coords:
            v.extend((c >> t) & 1 for t in range(k2))
        return tuple(v)

    # projective points of F_q^r: first nonzero coordinate normalized to 1
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
    expected = (2 ** n2 - 1) // (2 ** k2 - 1)
    assert len(points) == expected, (len(points), expected)

    spread = []
    for c in points:
        line = frozenset(to_bits(tuple(pmul(lam, cj, f) for cj in c))
                         for lam in range(q))
        assert len(line) == q
        spread.append(line)
    return spread


# --------------------------------------------------------------- loading
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


def pack(v, p):
    """Pack an element of U_N into a single int, for the pairwise index."""
    key = 0
    for b in v:
        key = (key << 1) | b
    for x in p:
        key = (key << 7) | x          # 7 bits per image: fine for N <= 127
    return key


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True,
                    help="spreadcompat_*.json: m disjoint spread-compatible "
                         "codewords of order 2^k in U_n")
    ap.add_argument("--n", type=int, required=True, help="BASE length n")
    ap.add_argument("--k", type=int, required=True, help="BASE k (order 2^k)")
    ap.add_argument("--out-tag", default=None,
                    help="tag for clique_<tag>_lift.json (default: <2n><4k><2k>)")
    a = ap.parse_args()

    n, k = a.n, a.k
    assert n % k == 0, "Proposition prodspread requires k | n"
    N, D, K = 2 * n, 4 * k, 2 * k
    order_base, order_lift = 2 ** k, 2 ** K
    tag = a.out_tag or f"{N}{D}{K}"

    e_n, idp_n, prod_n = make_group(n)
    e_N, idp_N, prod_N = make_group(N)
    zero_n, zero_N = tuple([0] * n), tuple([0] * N)
    assert check_associativity(n, prod_n), "U_n operation not associative"
    assert check_associativity(N, prod_N), "U_2n operation not associative"
    print(f"[check] group operations verified associative in U_{n} and U_{N}")

    # ---- 1. base codewords, re-verified from first principles ----
    t0 = time.time()
    base = load(a.base)
    m = len(base)
    print(f"\nbase: {m} codewords of order {order_base} from {a.base}")
    bad = []
    for i, H in enumerate(base):
        if not certify_subgroup(H, e_n, prod_n, order_base):
            bad.append((i, "not a subgroup of the right order"))
        elif len({v for v, p in H}) != order_base:
            bad.append((i, "degenerate"))
        elif any(p == idp_n for v, p in H if v != zero_n):
            bad.append((i, "not spread-compatible"))
    if bad:
        for i, why in bad[:10]:
            print(f"  FAIL base[{i}]: {why}")
        sys.exit(1)
    seen = {}
    for i, H in enumerate(base):
        for el in H:
            if el == e_n:
                continue
            if el in seen:
                print(f"  FAIL: base[{seen[el]}] and base[{i}] share an element")
                sys.exit(1)
            seen[el] = i
    print(f"  base verified: subgroups, nondegenerate, spread-compatible, "
          f"pairwise disjoint ({time.time()-t0:.0f}s)")

    # ---- 2. spread of F_2^{2n}, embedded as linear codewords ----
    t0 = time.time()
    spread = desarguesian_spread(N, K)
    s = len(spread)
    spread_cw = [frozenset((v, idp_N) for v in V) for V in spread]
    print(f"\nspread: {s} linear codewords of order {order_lift} in U_{N} "
          f"({time.time()-t0:.0f}s)")

    # ---- 3. diagonal lifts D_i = iota(C_i x C_i) ----
    t0 = time.time()

    def oplus(sg, tu):
        return tuple(sg) + tuple(n + tu[j] for j in range(n))

    lifts = []
    for H in base:
        Hl = list(H)
        D_i = frozenset((tuple(u) + tuple(v), oplus(sg, tu))
                        for (u, sg) in Hl for (v, tu) in Hl)
        lifts.append(D_i)
    print(f"lifts: {m} diagonal codewords D_i of order {order_lift} "
          f"({time.time()-t0:.0f}s)")

    family = spread_cw + lifts
    M = len(family)
    print(f"\nlifted family at ({N},{D},{K}): {s} + {m} = {M} codewords")

    # ---- 4a. per-codeword verification ----
    t0 = time.time()
    for i, H in enumerate(family):
        if len(H) != order_lift:
            print(f"  FAIL family[{i}]: order {len(H)} != {order_lift}"); sys.exit(1)
        if len({v for v, p in H}) != order_lift:
            print(f"  FAIL family[{i}]: degenerate"); sys.exit(1)
        if not certify_subgroup(H, e_N, prod_N, order_lift):
            print(f"  FAIL family[{i}]: not closed under the product"); sys.exit(1)
        if i >= s and any(p == tuple(range(N)) for v, p in H if v != zero_N):
            print(f"  FAIL D_{i-s}: not spread-compatible in U_{N}"); sys.exit(1)
    print(f"  per-codeword: closed subgroups, order {order_lift}, "
          f"nondegenerate; all D_i spread-compatible ({time.time()-t0:.0f}s)")

    # ---- 4b. pairwise trivial intersection, via a packed element index ----
    t0 = time.time()
    owner = {}
    ok = True
    for i, H in enumerate(family):
        for (v, p) in H:
            if (v, p) == e_N:
                continue
            key = pack(v, p)
            j = owner.get(key)
            if j is not None:
                print(f"  FAIL: codewords {j} and {i} share a nonidentity element")
                ok = False
            owner[key] = i
    if not ok:
        sys.exit(1)
    pairs = M * (M - 1) // 2
    print(f"  pairwise: all {pairs:,} pairs meet only in the identity "
          f"({time.time()-t0:.0f}s)")

    factor = (s + m) / s
    print(f"\n===== VERIFIED: A^P({N},{D},{K}) >= {s} + {m} = {M} "
          f"(A_q = {s}, factor {factor:.2f}x) =====")

    # ---- 5. output ----
    out_clique = f"clique_{tag}_lift.json"
    if os.path.exists(out_clique):
        print(f"  NOTE: {out_clique} already exists and will be overwritten")
    serialized = [[[list(v), list(p)] for v, p in sorted(H)] for H in family]
    # The "_lift" suffix is not cosmetic: lifting the (4,4,2) base produces
    # tag 884, which would otherwise overwrite clique_884.json, the computed
    # clique of the main theorem.
    json.dump(serialized, open(f"clique_{tag}_lift.json", "w"))
    json.dump({"base_case": f"({n},{2*k},{k})", "lift_case": f"({N},{D},{K})",
               "base_file": os.path.basename(a.base), "m": m,
               "spread_size": s, "A_P": M, "A_q": s,
               "operation": "as in the paper, eq:Un",
               "pairs_verified": pairs,
               "verification_subgroups": True, "verification_nondegenerate": True,
               "verification_distances": True,
               "all_lifts_spread_compatible": True},
              open(f"results_{tag}_lift.json", "w"), indent=2)
    print(f"  written: clique_{tag}_lift.json, results_{tag}_lift.json")
    print(f"  independent re-check: python verify_psc.py clique_{tag}_lift.json "
          f"--n {N} --d {D} --k {K}")


if __name__ == "__main__":
    main()
