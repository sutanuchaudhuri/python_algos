"""Benchmark: naive vs. vectorized vs. contiguous vs. C-SIMD matrix multiply.

Compares four strategies on square matrices:
  1. Matrix.multiply             - naive list-of-lists, textbook ijk triple loop.
  2. FastMatrix.multiply         - VECTORIZED: transpose B + sum(map(mul, ...)),
                                   inner loop runs in C  -> fastest in pure Python.
  3. FastMatrix.multiply_contiguous - contiguous array('d'), NumPy-style layout;
                                   slower here because array access boxes floats.
  4. SimdMatrix.multiply         - compiled C kernel (auto-vectorized SIMD) called
                                   via ctypes  -> the real NumPy-style win.

Run with:  make bench   (or)   python benchmark.py
"""

import random
import time

from matrix import Matrix
from fast_matrix import FastMatrix
from simd_matrix import SimdMatrix, is_available as simd_available


def random_rows(n: int) -> "list[list[float]]":
    return [[random.uniform(-1.0, 1.0) for _ in range(n)] for _ in range(n)]


def timed(label: str, func) -> float:
    start = time.perf_counter()
    func()
    elapsed = time.perf_counter() - start
    print("  {:<36} {:>9.4f} s".format(label, elapsed))
    return elapsed


def benchmark(n: int) -> None:
    print("\n=== {n}x{n} matrices ===".format(n=n))
    rows_a = random_rows(n)
    rows_b = random_rows(n)

    naive_a, naive_b = Matrix(rows_a), Matrix(rows_b)
    fast_a, fast_b = FastMatrix(rows_a), FastMatrix(rows_b)

    t_naive = timed("Matrix.multiply (naive ijk)", lambda: naive_a.multiply(naive_b))
    t_vec = timed("FastMatrix.multiply (vectorized)", lambda: fast_a.multiply(fast_b))
    t_cont = timed("FastMatrix.multiply_contiguous", lambda: fast_a.multiply_contiguous(fast_b))

    print("  {:<36} {:>8.2f}x".format("speedup (vectorized vs naive)", t_naive / t_vec))
    print("  {:<36} {:>8.2f}x".format("speedup (contiguous vs naive)", t_naive / t_cont))

    if simd_available():
        simd_a, simd_b = SimdMatrix(rows_a), SimdMatrix(rows_b)
        t_simd = timed("SimdMatrix.multiply (C SIMD)", lambda: simd_a.multiply(simd_b))
        print("  {:<36} {:>8.2f}x".format("speedup (C SIMD vs naive)", t_naive / t_simd))
    else:
        print("  (SimdMatrix skipped: run `make libmatmul` to build the C kernel)")


def correctness_check() -> None:
    print("=== Correctness check (all agree) ===")
    a = [[1, 2, 3], [4, 5, 6]]
    b = [[7, 8], [9, 10], [11, 12]]
    naive = Matrix(a).multiply(Matrix(b)).to_list()
    vec = FastMatrix(a).multiply(FastMatrix(b)).to_list()
    cont = FastMatrix(a).multiply_contiguous(FastMatrix(b)).to_list()
    print("  naive      :", naive)
    print("  vectorized :", [[int(v) for v in row] for row in vec])
    print("  contiguous :", [[int(v) for v in row] for row in cont])
    if simd_available():
        simd = SimdMatrix(a).multiply(SimdMatrix(b)).to_list()
        print("  C SIMD     :", [[int(v) for v in row] for row in simd])
    else:
        print("  C SIMD     : (not built)")


def main() -> None:
    random.seed(42)
    correctness_check()
    for n in (32, 64, 128, 256):
        benchmark(n)


if __name__ == "__main__":
    main()
