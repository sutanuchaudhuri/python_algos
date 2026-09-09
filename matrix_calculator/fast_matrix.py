"""Faster matrix multiplication that mimics NumPy's ideas, without NumPy.

NumPy owes its speed to two things:
  1. Contiguous memory  - the array is one flat, row-major block (C-order).
  2. Vectorization      - tight inner loops run in compiled C over that block,
                          often using SIMD, instead of the Python interpreter.

We reproduce both ideas with only the standard library and measure what really
helps in *pure* Python:

  * ``multiply``            - VECTORIZED: transpose B so every output cell is a
                             dot product of two sequential rows, then hand the
                             inner loop to the C-implemented ``sum``/``map`` with
                             ``operator.mul``. This is ~2x faster than the naive
                             triple loop because the hot loop leaves the
                             interpreter and runs in C.

  * ``multiply_contiguous`` - CONTIGUOUS: stores data in a flat ``array('d')``
                             (exactly NumPy's memory layout) and walks it with a
                             cache-friendly ``ikj`` loop. Instructive, but in pure
                             Python it is actually SLOWER: every ``array`` access
                             boxes a fresh ``float`` object, and that per-element
                             interpreter overhead dwarfs any cache benefit. The
                             cache/SIMD payoff only materialises once a C layer
                             (like NumPy) drives the loop.

Lesson: contiguous memory alone is not enough in pure Python; the win comes from
vectorizing, i.e. moving the inner loop out of the interpreter and into C.
"""

import operator
from array import array


class FastMatrixError(Exception):
    """Raised when a FastMatrix operation cannot be performed."""


class FastMatrix:
    """A dense matrix optimised for pure-Python multiplication."""

    __slots__ = ("rows", "cols", "_data")

    def __init__(self, data: "list[list[float]]") -> None:
        if not data or not data[0]:
            raise FastMatrixError("matrix must be a non-empty list of rows.")
        cols = len(data[0])
        for row in data:
            if len(row) != cols:
                raise FastMatrixError("All rows must have the same number of columns.")
        self.rows = len(data)
        self.cols = cols
        self._data = [list(map(float, row)) for row in data]

    @property
    def shape(self) -> "tuple[int, int]":
        return (self.rows, self.cols)

    def get(self, r: int, c: int) -> float:
        return self._data[r][c]

    def multiply(self, other: "FastMatrix") -> "FastMatrix":
        """Vectorized multiply: transpose B, then dot rows with C builtins."""
        if self.cols != other.rows:
            raise FastMatrixError(
                "Incompatible shapes for multiplication: "
                "{}x{} * {}x{}.".format(self.rows, self.cols, other.rows, other.cols)
            )

        mul = operator.mul
        # Transpose B once so each column becomes a sequential row.
        b_cols = [list(col) for col in zip(*other._data)]
        return FastMatrix(
            [
                [sum(map(mul, a_row, b_col)) for b_col in b_cols]
                for a_row in self._data
            ]
        )

    def multiply_contiguous(self, other: "FastMatrix") -> "FastMatrix":
        """Contiguous flat-buffer multiply (NumPy-style memory layout, ikj loop)."""
        if self.cols != other.rows:
            raise FastMatrixError(
                "Incompatible shapes for multiplication: "
                "{}x{} * {}x{}.".format(self.rows, self.cols, other.rows, other.cols)
            )

        m, n, p = self.rows, self.cols, other.cols
        a = array("d", [v for row in self._data for v in row])
        b = array("d", [v for row in other._data for v in row])
        c = array("d", bytes(8 * m * p))  # contiguous zero-filled result buffer

        for i in range(m):
            a_row = i * n
            c_row = i * p
            for k in range(n):
                a_ik = a[a_row + k]
                if a_ik == 0.0:
                    continue
                b_row = k * p
                for j in range(p):  # unit-stride sweep of B row k and C row i
                    c[c_row + j] += a_ik * b[b_row + j]

        return FastMatrix([list(c[i * p:(i + 1) * p]) for i in range(m)])

    def to_list(self) -> "list[list[float]]":
        return [list(row) for row in self._data]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FastMatrix):
            return NotImplemented
        return self._data == other._data

    def __repr__(self) -> str:
        return "FastMatrix(rows={}, cols={})".format(self.rows, self.cols)
