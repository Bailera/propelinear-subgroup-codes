"""
psc_gen.py -- Generation library for propelinear subgroup codes.
================================================================
The group operation is the one used in the paper (eq:Un):

    (x,p)(y,q) = (x + p(y), p*q)
    action       p(v)_i = v_{p^{-1}(i)}
    composition  (p*q)(i) = p(q(i))

SUBGROUP GENERATION -- five complementary sources, because no single one
covers the space and a pool built from one alone is systematically incomplete:

  1. LINEAR codewords, built explicitly from the subspaces of dimension k in
     reduced row echelon form. Sampling never produces them, since they
     require the identity permutation at every support vector.
  2. LATTICE WALK: start from {e} and adjoin one random element at a time.
     Reaches subgroups needing any number of generators.
  3. INVOLUTION WALK: a walk restricted to commuting involutions. This is the
     only practical route to the elementary abelian subgroups Z_2^k, which
     need k generators and which a search over random pairs or triples
     essentially never produces.
  4. MIXED-ORDER WALK: covers the intermediate values of the minimal number
     of generators.
  5. STRUCTURED: cyclic subgroups <a> and two-generated <a,b> with prescribed
     element orders.

DIAGNOSTIC: report_pool() prints the distribution of candidates by minimal
number of generators d(G) = log2 |G : Phi(G)| together with the count of
linear codewords. If the value d(G)=k is missing, or if there are no linear
codewords, the generation is biased and the pool is incomplete.
"""
import json, pickle
from itertools import permutations, combinations
import random, math, time
from collections import Counter, defaultdict


def make(n):
    """Return the operation table of U_n in the convention of the paper."""
    e = (tuple([0]*n), tuple(range(n)))
    idp = tuple(range(n))
    _allp = list(permutations(range(n))) if n <= 8 else None

    def perm_inv(p):
        s = [0]*n
        for i in range(n): s[p[i]] = i
        return tuple(s)

    def perm_act(p, v):
        q = perm_inv(p)
        return tuple(v[q[i]] for i in range(n))

    def perm_mul(p, q):
        return tuple(p[q[i]] for i in range(n))

    def prod(a, b):
        (x, p), (y, q) = a, b
        pv = perm_act(p, y)
        return (tuple((x[i]+pv[i]) % 2 for i in range(n)), perm_mul(p, q))

    def perm_2pow():
        """A permutation of S_n of 2-power order, built by partitioning
        {0..n-1} into cycles of length 1, 2, 4 or 8. Constructing them is
        essential: in S_10 only 0.26% of permutations are involutions, so
        drawing at random and filtering finds almost nothing."""
        pts = list(range(n)); random.shuffle(pts)
        p = [0]*n; i = 0
        while i < n:
            allowed = [L for L in (1, 2, 4, 8) if i + L <= n]
            L = random.choice(allowed)
            cycle = pts[i:i+L]
            for j in range(L):
                p[cycle[j]] = cycle[(j+1) % L]
            i += L
        return tuple(p)

    def rnd():
        """An element of U_n of 2-power order. If p has order 2^j then (v,p)
        has order dividing 2^{j+1} for every v, so it suffices to build p of
        2-power order."""
        v = tuple(random.randint(0, 1) for _ in range(n))
        return (v, perm_2pow())

    def rnd_free():
        """An element whose permutation is drawn uniformly, with no restriction
        on its order. Used only by the associativity check."""
        v = tuple(random.randint(0, 1) for _ in range(n))
        if _allp is not None:
            p = random.choice(_allp)
        else:
            pl = list(range(n)); random.shuffle(pl); p = tuple(pl)
        return (v, p)

    def element_order(a, cap=32):
        x = a; k = 1
        while x != e:
            x = prod(x, a); k += 1
            if k > cap: return 10**9
        return k

    def extend(H, g, cap):
        """<H, g>, or None if it exceeds cap."""
        S = set(H) | {g}; frontier = [g]
        while frontier:
            x = frontier.pop()
            for y in list(S):
                for pr in (prod(x, y), prod(y, x)):
                    if pr not in S:
                        S.add(pr); frontier.append(pr)
                        if len(S) > cap: return None
        return frozenset(S)

    def nondeg(H):
        return len({v for v, p in H}) == len(H)

    def is_linear(H):
        return all(p == idp for v, p in H)

    def min_generators(H):
        """Minimal number of generators d(G) = log2 |G : Phi(G)|, for a 2-group."""
        Hl = list(H); inv = {}
        for a in Hl:
            for b in Hl:
                if prod(a, b) == e: inv[a] = b; break
        gen = set(prod(prod(a, b), inv[prod(b, a)]) for a in Hl for b in Hl)
        gen |= set(prod(g, g) for g in Hl)
        Phi = {e}; frontier = list(gen)
        while frontier:
            x = frontier.pop()
            if x in Phi: continue
            Phi.add(x)
            for y in list(Phi):
                for pr in (prod(x, y), prod(y, x)):
                    if pr not in Phi: frontier.append(pr)
        return int(math.log2(len(H)//len(Phi)))

    def check_assoc(trials=5000):
        bad = 0
        for _ in range(trials):
            a, b, c = rnd_free(), rnd_free(), rnd_free()
            if prod(prod(a, b), c) != prod(a, prod(b, c)): bad += 1
        return bad

    return dict(n=n, e=e, idp=idp, prod=prod, perm_inv=perm_inv, rnd=rnd,
                rnd_free=rnd_free, perm_2pow=perm_2pow,
                element_order=element_order, extend=extend, nondeg=nondeg, is_linear=is_linear,
                min_generators=min_generators, check_assoc=check_assoc)


# ---------------------------------------------------------------- source 1
def subspaces_rref(n, k, cap=None, seed=0):
    """All subspaces of dimension k of F_2^n, via reduced row echelon form,
    one per subspace with no repetition. Samples if cap is exceeded."""
    rng = random.Random(seed)
    out = []
    for pivots in combinations(range(n), k):
        free_positions = [(r, c) for r in range(k) for c in range(n)
                  if c > pivots[r] and c not in pivots]
        for m in range(2**len(free_positions)):
            M = [[0]*n for _ in range(k)]
            for r in range(k): M[r][pivots[r]] = 1
            for i, (r, c) in enumerate(free_positions):
                if (m >> i) & 1: M[r][c] = 1
            S = {tuple([0]*n)}
            for mask in range(1, 2**k):
                v = [0]*n
                for r in range(k):
                    if (mask >> r) & 1:
                        v = [(v[j]+M[r][j]) % 2 for j in range(n)]
                S.add(tuple(v))
            out.append(frozenset(S))
    if cap is not None and len(out) > cap:
        out = rng.sample(out, cap)
    return out


def gauss_binom(n, k, q=2):
    """Number of subspaces of dimension k of F_q^n."""
    num = den = 1
    for i in range(k):
        num *= (q**(n-i) - 1)
        den *= (q**(k-i) - 1)
    return num // den


def subspaces_random(n, k, count, seed=0):
    """A sample of subspaces of dimension k, generated directly. Needed when
    the total is too large to enumerate (e.g. [10,5]_2 ~ 10^8): k independent
    vectors are drawn and their span is formed."""
    rng = random.Random(seed)
    zero = tuple([0]*n)
    out = set()
    attempts = 0
    while len(out) < count and attempts < count*60:
        attempts += 1
        basis = []
        span = {zero}
        for _ in range(k):
            free_positions = [v for v in (tuple(rng.randint(0, 1) for _ in range(n))
                                  for _ in range(30)) if v not in span]
            if not free_positions: break
            v = free_positions[0]
            basis.append(v)
            span = {tuple((a[i]+b[i]) % 2 for i in range(n))
                    for a in span for b in (zero, v)}
        if len(span) == 2**k:
            out.add(frozenset(span))
    return list(out)


def linear_codewords(G, k, cap=None, seed=0, enum_threshold=60000):
    """Linear codewords: subspaces of dimension k carrying the identity
    permutation. If the total number of subspaces exceeds enum_threshold they
    are sampled directly instead of being enumerated."""
    n, idp = G['n'], G['idp']
    total = gauss_binom(n, k)
    if total <= enum_threshold:
        S = subspaces_rref(n, k, cap=cap, seed=seed)
        print(f"    (subspaces of dim {k} in F_2^{n}: {total} in total, used {len(S)})")
    else:
        target_count = cap if cap else 20000
        S = subspaces_random(n, k, target_count, seed=seed)
        print(f"    (subspaces of dim {k} in F_2^{n}: {total} in total, "
              f"sampled {len(S)})")
    return [frozenset((v, idp) for v in Sub) for Sub in S]


# ---------------------------------------------------------------- source 2
def lattice_walk(G, target, tries=12):
    """Lattice walk: {e} -> ... -> a subgroup of order target."""
    e, extend, rnd = G['e'], G['extend'], G['rnd']
    H = frozenset([e])
    while len(H) < target:
        ok = False
        for _ in range(tries):
            g = rnd()
            if g in H: continue
            K = extend(H, g, target)
            if K is not None:
                H = K; ok = True; break
        if not ok: return None
    return H if len(H) == target else None


# ---------------------------------------------------------------- source 3
def collect_involutions(G, seconds=20, cap=40000):
    """Collect elements of order 2, for the elementary abelian subgroups."""
    element_order, rnd = G['element_order'], G['rnd']
    t0 = time.time(); out = []
    while time.time()-t0 < seconds and len(out) < cap:
        a = rnd()
        if element_order(a) == 2: out.append(a)
    return out


def involution_walk(G, target, inv2, sample_size=400):
    """A walk using ONLY commuting involutions: reaches Z_2^k and its kin."""
    e, prod, extend = G['e'], G['prod'], G['extend']
    if len(inv2) < 3: return None
    H = frozenset([e]); cands = inv2
    while len(H) < target:
        pool = random.sample(cands, min(sample_size, len(cands)))
        chosen_ext = None
        for g in pool:
            if g in H: continue
            if all(prod(g, h) == prod(h, g) for h in H):
                K = extend(H, g, target)
                if K is not None: chosen_ext = K; break
        if chosen_ext is None: return None
        H = chosen_ext
    return H if len(H) == target else None


def mixed_walk(G, target, buckets, sample_size=200):
    """A walk extending by elements of mixed order, not only involutions. It
    covers the intermediate values of d(G): involutions alone yield mostly
    elementary abelian subgroups (d=k) and generic elements mostly d=2, so the
    mixture fills the gap."""
    e, extend = G['e'], G['extend']
    available = [o for o in buckets if buckets[o]]
    if not available: return None
    H = frozenset([e])
    while len(H) < target:
        chosen_ext = None
        for _ in range(sample_size):
            o = random.choice(available)
            g = random.choice(buckets[o])
            if g in H: continue
            K = extend(H, g, target)
            if K is not None and K != H: chosen_ext = K; break
        if chosen_ext is None: return None
        H = chosen_ext
    return H if len(H) == target else None


# ---------------------------------------------------------------- sources 4-5
def structured(G, target, buckets):
    """Cyclic <a> and two-generated <a,b> with prescribed element orders."""
    e, extend = G['e'], G['extend']
    orders = sorted(buckets)
    if target in buckets and buckets[target] and random.random() < 0.4:
        return extend(frozenset([e]), random.choice(buckets[target]), target)
    available = [o for o in orders if buckets[o]]
    if len(available) < 2: return None
    o1 = random.choice(available); o2 = random.choice(available)
    H = extend(frozenset([e]), random.choice(buckets[o1]), target)
    if H is None: return None
    return extend(H, random.choice(buckets[o2]), target)


# ---------------------------------------------------------------- diagnostic
def report_pool(G, pool, k, label="pool", dgen_sample=3000, warn=True):
    """Print the distribution by d(G) and the count of linear codewords, and
    WARN if d(G)=k is missing. On large pools the d(G) distribution is
    estimated on a sample, since computing Phi(G) for hundreds of thousands of
    subgroups of order 32 would be prohibitive."""
    min_generators, is_linear = G['min_generators'], G['is_linear']
    sub = pool if len(pool) <= dgen_sample else random.sample(list(pool), dgen_sample)
    c = Counter(min_generators(H) for H in sub)
    if len(sub) < len(pool):
        label = f"{label}, d(G) on a sample of {len(sub)}"
    lin = sum(1 for H in pool if is_linear(H))
    print(f"  [{label}] {len(pool)} subgroups | d(G): {dict(sorted(c.items()))} "
          f"| linear: {lin}")
    if not warn:
        # In a CLIQUE, having no linear codewords is the result being sought
        # (cf. the section on linear codewords), not a defect.
        return True
    warnings_ = []
    if c.get(k, 0) == 0:
        warnings_.append(f"NO subgroups with d(G)={k} (elementary abelian Z_2^{k})")
    if lin == 0:
        warnings_.append("NO linear codewords")
    for a in warnings_:
        print(f"  *** WARNING: {a} -> the generation is biased and the pool is incomplete")
    return len(warnings_) == 0


# ---------------------------------------------------------------- pipeline
def pipeline(n, d, k, paper_old, a_q, GEN_TIME=300, ILP_TIME=900,
             LIN_CAP=20000, GREEDY_REPS=5000, seed=0, PREVIOUS_POOL=None,
             MAX_ILP_VARS=400000, MAX_ILP_PAIRWISE=8000, TAG=""):
    """Full pipeline for one case (n,d,k): generation -> pool -> greedy -> ILP.
    Writes pool_<ndk>.pkl, clique_<ndk>.pkl/.json and results_<ndk>.json."""
    import pulp
    random.seed(seed)
    ORDER = 2**k
    max_shared = 2**((2*k-d)//2) - 1
    NAME = f"{n}{d}{k}{TAG}"   # optional suffix for running parallel instances
    G = make(n)
    e, prod, idp = G['e'], G['prod'], G['idp']

    bad = G['check_assoc']()
    assert bad == 0, f"ABORTED: the group operation is not associative ({bad}/5000)"
    print(f"[check] operation verified associative (0/5000)")
    print(f"Case ({n},{d},{k}): order {ORDER}, threshold |core cap core| <= {max_shared}\n")

    pool = set()

    # --- source 0: previous pool (optional). Everything is revalidated before merging ---
    if PREVIOUS_POOL:
        import os
        paths = [PREVIOUS_POOL] if isinstance(PREVIOUS_POOL, str) else list(PREVIOUS_POOL)
        for path in paths:
            if not os.path.exists(path):
                print(f"  source 0: {path} does not exist, skipped"); continue
            data = pickle.load(open(path, "rb"))
            good = 0; bad = 0; n_before_r = len(pool)
            for H in data:
                Hf = frozenset((tuple(v), tuple(p)) for v, p in H)
                if len(Hf) != ORDER or not G['nondeg'](Hf):
                    bad += 1; continue
                if all(prod(a, b) in Hf for a in Hf for b in Hf):
                    pool.add(Hf); good += 1
                else:
                    bad += 1
            print(f"  source 0 ({path}): {good} valid, +{len(pool)-n_before_r} new"
                  + (f", {bad} discarded" if bad else ""))

    # --- source 1: explicit linear codewords ---
    t0 = time.time()
    lin = linear_codewords(G, k, cap=LIN_CAP, seed=seed)
    n_before = len(pool)
    pool |= set(H for H in lin if G['nondeg'](H))
    print(f"  source 1 (explicit linear codewords): +{len(pool)-n_before} ({time.time()-t0:.0f}s)")

    # --- buckets by element_order, for sources 3, 4 and 5 ---
    t0 = time.time()
    buckets = {2**i: [] for i in range(1, k+1)}
    inv2 = []
    while time.time()-t0 < GEN_TIME*0.15:
        a = G['rnd']()
        o = G['element_order'](a)
        if o in buckets and len(buckets[o]) < 40000: buckets[o].append(a)
        if o == 2 and len(inv2) < 40000: inv2.append(a)
    print(f"  reserves: " + " ".join(f"ord{o}:{len(buckets[o])}" for o in sorted(buckets)))

    # --- sources 2, 3, 4, 5 mixed ---
    def accept(H):
        if H and len(H) == ORDER and G['nondeg'](H):
            pool.add(H); return True
        return False

    t0 = time.time(); n0 = len(pool)
    while time.time()-t0 < GEN_TIME*0.85:
        r = random.random()
        # mixture reweighted towards the most productive sources: in U_10 the
        # involution walk yields ~10x more than the lattice walk.
        # Involutions and mixed are favoured (they cover Z_2^k and the intermediate
        # d(G), the bulk of the subgroups) with small shares left to the lattice
        # walk and the structured source, so as not to lose the types only they
        # reach.
        if r < 0.55:
            accept(involution_walk(G, ORDER, inv2))
        elif r < 0.85:
            accept(mixed_walk(G, ORDER, buckets))
        elif r < 0.93:
            accept(lattice_walk(G, ORDER))
        else:
            accept(structured(G, ORDER, buckets))
    print(f"  sources 2-5 (lattice/involution/mixed/structured): +{len(pool)-n0} "
          f"({time.time()-t0:.0f}s)")

    pool = list(pool)
    print(f"\nTOTAL POOL: {len(pool)} nondegenerate subgroups of order {ORDER}")
    healthy = report_pool(G, pool, k, "pool")
    pickle.dump([[[list(v), list(p)] for v, p in H] for H in pool],
                open(f"pool_{NAME}.pkl", "wb"))

    # --- greedy ---
    core = [H - {e} for H in pool]
    N = len(pool)
    def greedy(s):
        random.seed(s); order_idx = list(range(N)); random.shuffle(order_idx)
        if max_shared == 0:
            used = set(); changed = []
            for i in order_idx:
                if core[i] & used: continue
                changed.append(i); used |= core[i]
            return changed
        # element -> already chosen positions: avoids comparing the
        # candidate against the whole clique at each step (unusable with
        # pools of hundreds of thousands).
        occupied = defaultdict(list); changed = []
        for i in order_idx:
            shares = Counter()
            for el in core[i]:
                for pos in occupied.get(el, ()): shares[pos] += 1
            if shares and max(shares.values()) > max_shared: continue
            pos = len(changed); changed.append(i)
            for el in core[i]: occupied[el].append(pos)
        return changed
    reps = GREEDY_REPS if max_shared == 0 else max(200, GREEDY_REPS//20)
    best = []
    for s in range(reps):
        c = greedy(s)
        if len(c) > len(best): best = c
    print(f"  greedy ({reps} restarts): {len(best)}")

    # --- ILP ---
    # With many variables PuLP builds the model in Python, writes a huge .lp
    # and CBC parses it, all BEFORE timeLimit starts counting (the limit only
    # governs branch-and-bound). The process looks hung for hours. Above the
    # cap the ILP is skipped and the greedy clique is kept; the ILP is then
    # run separately with ilp_subpool.py, which restricts the problem to a
    # manageable subpool.
    # For d < 2k (max_shared > 0) the constraints cannot be built through an
    # element index: every pair must be intersected, which is O(N^2) and runs
    # BEFORE any warning could be printed. The cap is therefore much lower in
    # that branch.
    ilp_cap = MAX_ILP_VARS if max_shared == 0 else MAX_ILP_PAIRWISE
    if N > ilp_cap:
        print(f"\n  ILP SKIPPED: {N} candidates exceeds the cap of {ilp_cap}"
              + ("" if max_shared == 0 else
                 f" (sub-maximum distance: pairwise construction is O(N^2))") + ".")
        print(f"  To exploit the pool, run separately:")
        print(f"    python ilp_subpool.py --pool pool_{n}{d}{k}.pkl "
              f"--n {n} --d {d} --k {k} --aq {a_q} --seed clique_{n}{d}{k}.pkl")
        chosen, method = best, "greedy (ILP skipped due to size; use ilp_subpool.py)"
        value = len(chosen)
        print(f"\n===== A^P({n},{d},{k}) >= {value}  [{method}] =====")
        clique = [pool[i] for i in chosen]
        cc = [H - {e} for H in clique]
        ok_dist = all(len(cc[p] & cc[q]) <= max_shared for p in range(value) for q in range(p+1, value))
        ok_group = all(all(prod(u, w) in H for u in H for w in H) for H in clique)
        n_linear = sum(1 for H in clique if G['is_linear'](H))
        print(f"  verification: distances={ok_dist} groups={ok_group} linear={n_linear}")
        print(f"  factor {value/a_q:.1f}x  (previous paper value: {paper_old}, A_q={a_q})")
        report_pool(G, clique, k, "clique", warn=False)
        serialized = [[[list(v), list(p)] for v, p in H] for H in clique]
        pickle.dump(serialized, open(f"clique_{NAME}.pkl", "wb"))
        json.dump(serialized, open(f"clique_{NAME}.json", "w"))
        json.dump({"case": f"({n},{d},{k})", "A_P": value, "method": method,
                   "operation": "as in the paper, eq:Un", "pool_size": N,
                   "pool_healthy": healthy, "verification_distances": ok_dist,
                   "verification_groups": ok_group, "linear": n_linear,
                   "nonlinear": value-n_linear, "previous_paper_value": paper_old,
                   "A_q": a_q, "note": "lower bound; ILP still to be run (ilp_subpool.py)"},
                  open(f"results_{NAME}.json", "w"), indent=2, ensure_ascii=False)
        print(f"  written: results_{NAME}.json, clique_{NAME}.pkl, pool_{NAME}.pkl")
        return value

    print(f"  building the ILP...")
    prob = pulp.LpProblem("c", pulp.LpMaximize)
    x = [pulp.LpVariable(f"x{i}", cat="Binary") for i in range(N)]
    prob += pulp.lpSum(x)
    n_constraints = 0
    if max_shared == 0:
        from collections import defaultdict
        elem_to_cands = defaultdict(list)
        for i, c in enumerate(core):
            for el in c: elem_to_cands[el].append(i)
        for owners in elem_to_cands.values():
            if len(owners) >= 2:
                prob += pulp.lpSum(x[i] for i in owners) <= 1; n_constraints += 1
    else:
        for i in range(N):
            for j in range(i+1, N):
                if len(core[i] & core[j]) > max_shared:
                    prob += x[i]+x[j] <= 1; n_constraints += 1
    density = n_constraints / max(1, N)
    print(f"  {n_constraints} constraints over {N} variables (density {density:.2f})")
    if density > 0.15:
        print(f"  WARNING: dense conflict graph; CBC may take very long. If it does,")
        print(f"         stop it and use ilp_subpool.py on pool_{n}{d}{k}.pkl")
    print(f"  solving (limit {ILP_TIME}s)...")
    t0 = time.time()
    try:
        # msg=1: without CBC's output a long ILP is indistinguishable from a hang
        prob.solve(pulp.PULP_CBC_CMD(msg=1, timeLimit=ILP_TIME))
        status = pulp.LpStatus[prob.status]
        selected = [i for i in range(N) if pulp.value(x[i]) and pulp.value(x[i]) > 0.5]
    except Exception as ex:
        print(f"  (CBC failed: {ex})"); selected = []; status = "CBC error"
    def is_valid(s):
        return all(len(core[s[a]] & core[s[b]]) <= max_shared
                   for a in range(len(s)) for b in range(a+1, len(s)))
    if len(selected) > len(best) and is_valid(selected):
        chosen, method = selected, f"ILP ({status})"
    else:
        chosen, method = best, f"greedy (ILP {status} did not improve)"
    value = len(chosen)
    print(f"\n===== A^P({n},{d},{k}) = {value}  [{method}]  ({time.time()-t0:.0f}s) =====")

    # --- verification and output ---
    clique = [pool[i] for i in chosen]
    cc = [H - {e} for H in clique]
    ok_dist = all(len(cc[a] & cc[b]) <= max_shared for a in range(value) for b in range(a+1, value))
    ok_group = all(all(prod(a, b) in H for a in H for b in H) for H in clique)
    n_linear = sum(1 for H in clique if G['is_linear'](H))
    print(f"  verification: distances={ok_dist} groups={ok_group} linear={n_linear}")
    print(f"  factor {value/a_q:.1f}x  (previous paper value: {paper_old}, A_q={a_q})")
    report_pool(G, clique, k, "clique", warn=False)
    serialized = [[[list(v), list(p)] for v, p in H] for H in clique]
    pickle.dump(serialized, open(f"clique_{NAME}.pkl", "wb"))
    json.dump(serialized, open(f"clique_{NAME}.json", "w"))
    json.dump({"case": f"({n},{d},{k})", "A_P": value, "method": method,
               "operation": "as in the paper, eq:Un", "pool_size": N,
               "pool_healthy": healthy, "verification_distances": ok_dist,
               "verification_groups": ok_group, "linear": n_linear,
               "nonlinear": value-n_linear, "previous_paper_value": paper_old,
               "A_q": a_q, "note": "lower bound (sampled pool); raise GEN_TIME for a larger pool"},
              open(f"results_{NAME}.json", "w"), indent=2, ensure_ascii=False)
    print(f"  written: results_{NAME}.json, clique_{NAME}.pkl, pool_{NAME}.pkl")
    return value
