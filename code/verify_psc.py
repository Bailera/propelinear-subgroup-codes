#!/usr/bin/env python3
"""
verify_psc.py -- Independent verification of propelinear subgroup codes.

Checks that a clique file describes a valid propelinear subgroup code for the
parameters (n, d, k), verifying three properties from first principles:

  1. every codeword is a subgroup of U_n of order 2^k (closure under the
     twisted product, recomputed here rather than trusted);
  2. every codeword is nondegenerate, i.e. |supp(C)| = |C|;
  3. no two codewords share more than the permitted number of nonidentity
     elements, so that the minimum propelinear distance is at least d.

If all three hold, the number of codewords in the file is a certified lower
bound on A^P_nd(n, d, k), and the script reports it.

The group operation is recomputed independently of the generation code:

    (x, p)(y, q) = (x + p(y), p * q),
    (p(v))_i = v_{p^{-1}(i)},   (p * q)(i) = p(q(i)).

Distance and threshold. For subgroups C_i, C_j of order 2^k,

    d_P(C_i, C_j) = 2k - 2 log2 |C_i cap C_j|,

where the intersection is taken in U_n and contains the identity. Since the
identity lies in every subgroup, d_P(C_i, C_j) >= d is equivalent to a bound
on the number of shared NONIDENTITY elements,

    |C_i cap C_j| - 1 <= 2^((2k-d)/2) - 1,

which is the form checked below.

For d = 2k this threshold is 0 (codewords meet only in the identity); for
d < 2k it is positive, and sharing that many elements is allowed.

Usage:
    python verify_psc.py cliques/clique_663.json --n 6 --d 6 --k 3
    python verify_psc.py cliques/clique_643.json --n 6 --d 4 --k 3
    python verify_psc.py cliques/*.json            # parameters read from names

File names of the form clique_<n><d><k>.json are parsed when unambiguous
(663, 10105, 202010); otherwise pass --n --d --k explicitly.

Exit status is 0 if all checks pass and 1 otherwise, so the script can be used
in an automated test suite.
"""
import argparse
import itertools
import json
import os
import pickle
import re
import sys
from collections import defaultdict


# --------------------------------------------------------------- group setup
def make_group(n):
    """Return the operations of U_n = F_2^n : S_n with the paper's convention."""
    identity_perm = tuple(range(n))
    identity = (tuple([0] * n), identity_perm)

    def perm_inverse(p):
        s = [0] * n
        for i in range(n):
            s[p[i]] = i
        return tuple(s)

    def perm_action(p, v):
        """(p(v))_i = v_{p^{-1}(i)}."""
        q = perm_inverse(p)
        return tuple(v[q[i]] for i in range(n))

    def perm_compose(p, q):
        """(p * q)(i) = p(q(i))."""
        return tuple(p[q[i]] for i in range(n))

    def product(a, b):
        (x, p), (y, q) = a, b
        py = perm_action(p, y)
        return (tuple((x[i] + py[i]) % 2 for i in range(n)), perm_compose(p, q))

    return identity, identity_perm, product


def check_associativity(n, product, trials=5000, seed=12345):
    """Sanity check: the operation used here must be associative."""
    import random
    rng = random.Random(seed)
    perms = list(itertools.permutations(range(n))) if n <= 8 else None

    def rand():
        v = tuple(rng.randint(0, 1) for _ in range(n))
        if perms is not None:
            p = rng.choice(perms)
        else:
            pl = list(range(n))
            rng.shuffle(pl)
            p = tuple(pl)
        return (v, p)

    for _ in range(trials):
        a, b, c = rand(), rand(), rand()
        if product(product(a, b), c) != product(a, product(b, c)):
            return False
    return True


# --------------------------------------------------------------- file loading
def load_clique(path):
    """Load a clique from .json or .pkl into a list of frozensets."""
    if path.endswith(".pkl"):
        with open(path, "rb") as fh:
            data = pickle.load(fh)
    else:
        with open(path) as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            for field in ("codes", "candidates", "clique", "pool"):
                if field in data:
                    data = data[field]
                    break
    return [frozenset((tuple(v), tuple(p)) for v, p in H) for H in data]


def infer_parameters(path):
    """Infer (n, d, k) from a file name such as clique_663.json or clique_10105.json."""
    m = re.search(r"(\d+)", os.path.basename(path))
    if not m:
        return None
    digits = m.group(1)
    if len(digits) == 3:                      # 663    -> (6, 6, 3)
        return int(digits[0]), int(digits[1]), int(digits[2])
    if len(digits) == 4:                      # no unambiguous split
        return None                           # pass --n --d --k explicitly
    if len(digits) == 5:                      # 10105 -> (10,10,5), 12126 -> (12,12,6)
        n_, d_, k_ = int(digits[:2]), int(digits[2:4]), int(digits[4])
        if d_ == 2 * k_ and n_ == d_:
            return n_, d_, k_
        return int(digits[0]), int(digits[1:3]), int(digits[3:])
    if len(digits) == 6:                      # 202010 -> (20,20,10)
        for split in ((2, 4), (2, 3)):
            n_ = int(digits[:split[0]])
            d_ = int(digits[split[0]:split[1]])
            k_ = int(digits[split[1]:])
            if n_ == d_ == 2 * k_:
                return n_, d_, k_
        return None
    return None


# --------------------------------------------------------------- verification
def verify(path, n, d, k, verbose=True):
    identity, identity_perm, product = make_group(n)
    order = 2 ** k
    max_shared = 2 ** ((2 * k - d) // 2) - 1

    if verbose:
        print(f"\n=== {os.path.basename(path)} : parameters (n,d,k) = ({n},{d},{k}) ===")
        print(f"    codeword order 2^{k} = {order}, "
              f"allowed |C_i* cap C_j*| <= {max_shared}")

    if not check_associativity(n, product):
        print("    FAIL: the group operation is not associative (internal error)")
        return False

    clique = load_clique(path)
    M = len(clique)
    failures = []

    # 1. order
    wrong_order = [i for i, H in enumerate(clique) if len(H) != order]
    if wrong_order:
        failures.append(f"{len(wrong_order)} codewords do not have order {order}")

    # 2. nondegeneracy
    degenerate = [i for i, H in enumerate(clique)
                  if len({v for v, p in H}) != len(H)]
    if degenerate:
        failures.append(f"{len(degenerate)} codewords are degenerate")

    # 3. closure under the twisted product
    not_closed = []
    for i, H in enumerate(clique):
        if any(product(a, b) not in H for a in H for b in H):
            not_closed.append(i)
    if not_closed:
        failures.append(f"{len(not_closed)} codewords are not closed under the product")

    # 4. pairwise distance, via an element -> codewords index
    cores = [H - {identity} for H in clique]
    element_to_codewords = defaultdict(list)
    for i, core in enumerate(cores):
        for element in core:
            element_to_codewords[element].append(i)
    shared = defaultdict(int)
    for owners in element_to_codewords.values():
        for a in range(len(owners)):
            for b in range(a + 1, len(owners)):
                shared[(owners[a], owners[b])] += 1
    violations = [pair for pair, count in shared.items() if count > max_shared]
    if violations:
        failures.append(f"{len(violations)} pairs exceed the allowed intersection")

    if verbose:
        print(f"    codewords ............... {M}")
        print(f"    order 2^{k} .............. {'OK' if not wrong_order else 'FAIL'}")
        print(f"    nondegenerate ........... {'OK' if not degenerate else 'FAIL'}")
        print(f"    closed under product .... {'OK' if not not_closed else 'FAIL'}")
        print(f"    pairwise distance >= {d} .. {'OK' if not violations else 'FAIL'}")
        if shared:
            distribution = defaultdict(int)
            for count in shared.values():
                distribution[count] += 1
            summary = ", ".join(f"{c} element(s): {n_}"
                                for c, n_ in sorted(distribution.items()))
            print(f"    intersection profile .... {summary}")
        linear = sum(1 for H in clique if all(p == identity_perm for v, p in H))
        print(f"    linear codewords ........ {linear}")

    if failures:
        for f in failures:
            print(f"    FAIL: {f}")
        return False
    if verbose:
        print(f"    VERIFIED: A^P_nd({n},{d},{k}) >= {M}")
    return True


def main():
    ap = argparse.ArgumentParser(
        description="Verify propelinear subgroup codes from a clique file.")
    ap.add_argument("files", nargs="+", help="clique files (.json or .pkl)")
    ap.add_argument("--n", type=int, help="length (inferred from file name if omitted)")
    ap.add_argument("--d", type=int, help="minimum distance")
    ap.add_argument("--k", type=int, help="log2 of the codeword order")
    ap.add_argument("--quiet", action="store_true", help="only print the final verdict")
    args = ap.parse_args()

    # Expand wildcards here rather than relying on the shell: bash and zsh
    # expand cliques/*.json before Python sees it, but Windows PowerShell and
    # cmd pass the pattern through literally.
    import glob
    files = []
    for pattern in args.files:
        matches = sorted(glob.glob(pattern))
        files.extend(matches if matches else [pattern])

    all_ok = True
    for path in files:
        if args.n and args.d and args.k:
            n, d, k = args.n, args.d, args.k
        else:
            inferred = infer_parameters(path)
            if inferred is None:
                print(f"{path}: cannot infer (n,d,k); pass --n --d --k explicitly")
                all_ok = False
                continue
            n, d, k = inferred
        all_ok &= verify(path, n, d, k, verbose=not args.quiet)

    print("\n" + ("ALL FILES VERIFIED" if all_ok else "VERIFICATION FAILED"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
