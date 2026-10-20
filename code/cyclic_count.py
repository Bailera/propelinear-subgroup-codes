"""
cyclic_count.py -- check of the theorem on cyclic codewords (Section 5.5).

For h = 2^(k-1), the paper proves that the largest code with parameters (n, 2k, k)
all of whose codewords are cyclic has exactly

    J_k(n) = sum_{m >= 0} N(n, m h) * (2^(m h) * W_h(n - 2 m h) - 1)

codewords, where N(n, j) = n! / ((n-2j)! 2^j j!) counts the involutions of S_n with
j transpositions (identity included for j = 0) and W_h(f) counts the vectors of
F_2^f whose weight is a multiple of h.

This script
  1. evaluates J_k(n) by the formula;
  2. recomputes it by brute force for n <= 8: it lists every element g = (x, sigma)
     of U_n with sigma^(2h) = id, computes g^h, and counts the distinct involutions
     with nonzero vector part obtained in this way;
  3. prints the ratio J_k(n) / iota(n), which tends to 4^(1-k) (Corollary in 5.5).

Convention (as in the rest of the repository): sigma acts by w[sigma[i]] = v[i], and
(x, s)(y, t) = (x + s(y), s o t) with (s o t)[i] = s[t[i]].
Requires numpy. Runs in about a minute.
"""
import itertools
import math
from fractions import Fraction

import numpy as np


def N(n, j):
    if 2 * j > n:
        return 0
    return math.factorial(n) // (math.factorial(n - 2 * j) * 2**j * math.factorial(j))


def W(h, f):
    if h == 1:
        return 2**f
    return sum(math.comb(f, i) for i in range(0, f + 1, h))


def J(k, n):
    h = 2**(k - 1)
    return sum(N(n, m * h) * (2**(m * h) * W(h, n - 2 * m * h) - 1)
               for m in range(n // (2 * h) + 1))


def iota(n):
    return J(1, n)  # J_1(n) is the involution count of Proposition 4.2


def brute(k, n):
    """Number of involutions (s, tau) with s != 0 that are h-th powers of an element of order 2^k."""
    h = 2**(k - 1)
    X = np.arange(1 << n, dtype=np.int64)
    bits = (X[:, None] >> np.arange(n)) & 1
    ident = tuple(range(n))

    def act_all(p):  # y[x] = p(x) for every x, as bitmasks
        w = np.zeros_like(X)
        for i in range(n):
            w |= bits[:, i] << p[i]
        return w

    def comp(s, t):
        return tuple(s[t[i]] for i in range(n))

    found = set()
    for p in itertools.permutations(range(n)):
        q = ident
        for _ in range(2 * h):
            q = comp(p, q)
        if q != ident:          # need sigma^(2h) = id
            continue
        s = np.zeros_like(X)    # s_h = sum_{u<h} sigma^u(x)
        pu = ident
        for _ in range(h):
            s ^= X if pu == ident else act_all(pu)
            pu = comp(p, pu)
        tau = pu                # sigma^h
        tau_s = s if tau == ident else act_all(tau)[s]
        ok = (s != 0) & (tau_s == s)   # (s, tau) is an involution with s != 0
        for sv in np.unique(s[ok]):
            found.add((int(sv), tau))
    return len(found)


if __name__ == '__main__':
    print('Formula against brute force (n = 4..8, k = 2, 3, 4):')
    all_ok = True
    for n in range(4, 9):
        row = []
        for k in (2, 3, 4):
            b, f = brute(k, n), J(k, n)
            all_ok &= (b == f)
            row.append(f'k={k}: {f}' + ('' if b == f else f' (brute force {b})'))
        print(f'  n={n}:  ' + '   '.join(row))
    print('  all agree' if all_ok else '  MISMATCH')
    print('\nJ_k(2k) for k = 2..6:', [J(k, 2 * k) for k in range(2, 7)])
    print('\nJ_k(n)/iota(n)  (limits 1/4, 1/16, 1/64):')
    for n in (8, 16, 30, 50, 100, 200, 300):
        print(f'  n={n:<4}' + '  '.join(f'{float(Fraction(J(k, n), iota(n))):.4f}' for k in (2, 3, 4)))
