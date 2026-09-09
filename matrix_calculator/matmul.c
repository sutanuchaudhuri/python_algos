/* C-level SIMD matrix multiplication kernel for the pure-Python matrix calculator.
 *
 * This is the "real" version of what makes NumPy fast: a tight inner loop
 * compiled to native code. Built with -O3 -march=native -ffast-math, the
 * innermost unit-stride loop below is auto-vectorized by the compiler into
 * SIMD instructions (NEON on Apple Silicon / arm64, AVX on x86-64), processing
 * multiple doubles per instruction.
 *
 * Layout: row-major (C-order) contiguous double buffers, exactly like NumPy.
 * The `restrict` qualifiers promise the buffers do not alias, which lets the
 * compiler vectorize freely.
 */

#include <string.h>   /* memset: fill a block of memory with a byte value */
#include <stddef.h>   /* size_t: the unsigned type used for sizes / offsets */

/* matmul: C(m x p) = A(m x n) * B(n x p), all row-major contiguous doubles.
 *
 * Parameters
 *   A : pointer to the first element of matrix A, laid out row-by-row.
 *   B : pointer to the first element of matrix B, laid out row-by-row.
 *   C : pointer to the output buffer (m * p doubles); overwritten by this call.
 *   m : number of rows in A (and in C).
 *   n : number of columns in A and rows in B (the shared/contraction dimension).
 *   p : number of columns in B (and in C).
 *
 * Memory model
 *   Every matrix is a single flat block of doubles in row-major (C) order, so
 *   element (r, c) of an X-by-Y matrix lives at index r*Y + c. There are no
 *   per-row objects or pointers-to-rows: it is one contiguous array, exactly
 *   like a NumPy array's data buffer.
 *
 * `restrict`
 *   Promises the compiler that A, B and C do not overlap in memory. Without it
 *   the compiler must assume a write to C[...] might change A or B, which
 *   blocks vectorization. With it, the inner loop is free to run in SIMD.
 *
 * Loop order (ikj)
 *   The classic textbook order is ijk, whose inner loop walks DOWN a column of
 *   B (stride p) - cache-unfriendly and hard to vectorize. Swapping to ikj makes
 *   the inner j-loop sweep ACROSS row k of B and row i of C with stride 1 (unit
 *   stride). Unit-stride sequential access is exactly the pattern the compiler
 *   turns into SIMD instructions and the CPU prefetcher loves. */
void matmul(const double *restrict A,
            const double *restrict B,
            double *restrict C,
            int m, int n, int p)
{
    /* Zero the whole output buffer up front. The ikj algorithm ACCUMULATES into
     * C (c_row[j] += ...), so each cell must start at 0. memset writes bytes,
     * and the all-zero bit pattern is exactly 0.0 for IEEE-754 doubles, so a
     * single memset correctly clears m*p doubles in one fast, often-vectorized
     * library call. The size is computed in size_t to avoid int overflow for
     * large matrices. */
    memset(C, 0, (size_t)m * (size_t)p * sizeof(double));

    for (int i = 0; i < m; ++i) {
        /* Cache the base pointers of row i of A and row i of C once per i. */
        const double *a_row = A + (size_t)i * n;
        double *c_row = C + (size_t)i * p;

        for (int k = 0; k < n; ++k) {
            /* One scalar from A, reused across the entire inner loop. Hoisting
             * it here avoids re-reading A on every j iteration. */
            const double a_ik = a_row[k];
            const double *b_row = B + (size_t)k * p;

            /* Inner kernel: c_row[j] += a_ik * b_row[j] for all j.
             * Both c_row and b_row are swept with stride 1, so the compiler
             * emits a fused multiply-add over several doubles per SIMD lane.
             * The pragma is a hint to Clang to vectorize and interleave (unroll
             * across vector registers) this loop. */
#pragma clang loop vectorize(enable) interleave(enable)
            for (int j = 0; j < p; ++j) {
                c_row[j] += a_ik * b_row[j];
            }
        }
    }
}
