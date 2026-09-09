"""Matrix calculator built from scratch using only the Python standard library.

Uses plain lists (no NumPy, no advanced data structures) and OOP principles.
Matrix multiplication is the primary operation; addition, subtraction, scalar
multiplication and transpose are also provided.
"""


class MatrixError(Exception):
    """Raised when a matrix operation cannot be performed."""


Number = (int, float)


class Matrix:
    """A 2D matrix backed by a list of lists of numbers."""

    def __init__(self, rows: "list[list[int | float]]") -> None:
        if not isinstance(rows, list) or len(rows) == 0:
            raise MatrixError("Matrix must be a non-empty list of rows.")

        width: "int | None" = None
        clean_rows: "list[list[int | float]]" = []
        for row in rows:
            if not isinstance(row, list) or len(row) == 0:
                raise MatrixError("Each row must be a non-empty list of numbers.")
            if width is None:
                width = len(row)
            elif len(row) != width:
                raise MatrixError("All rows must have the same number of columns.")

            clean_row = []
            for value in row:
                if isinstance(value, bool) or not isinstance(value, Number):
                    raise MatrixError("Matrix values must be int or float.")
                clean_row.append(value)
            clean_rows.append(clean_row)

        self._rows = clean_rows
        self._num_rows = len(clean_rows)
        self._num_cols = width

    @property
    def num_rows(self) -> int:
        return self._num_rows

    @property
    def num_cols(self) -> int:
        return self._num_cols

    @property
    def shape(self) -> "tuple[int, int]":
        return (self._num_rows, self._num_cols)

    def get(self, r: int, c: int) -> "int | float":
        self._check_index(r, c)
        return self._rows[r][c]

    def set(self, r: int, c: int, value: "int | float") -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise MatrixError("Matrix values must be int or float.")
        self._check_index(r, c)
        self._rows[r][c] = value

    def _check_index(self, r: int, c: int) -> None:
        if not (0 <= r < self._num_rows) or not (0 <= c < self._num_cols):
            raise MatrixError("Index out of range.")

    def add(self, other: "Matrix") -> "Matrix":
        self._require_same_shape(other, "addition")
        result = []
        for i in range(self._num_rows):
            result.append(
                [self._rows[i][j] + other._rows[i][j] for j in range(self._num_cols)]
            )
        return Matrix(result)

    def subtract(self, other: "Matrix") -> "Matrix":
        self._require_same_shape(other, "subtraction")
        result = []
        for i in range(self._num_rows):
            result.append(
                [self._rows[i][j] - other._rows[i][j] for j in range(self._num_cols)]
            )
        return Matrix(result)

    def scalar_multiply(self, scalar: "int | float") -> "Matrix":
        if isinstance(scalar, bool) or not isinstance(scalar, (int, float)):
            raise MatrixError("Scalar must be int or float.")
        result = []
        for i in range(self._num_rows):
            result.append([value * scalar for value in self._rows[i]])
        return Matrix(result)

    def multiply(self, other: "Matrix") -> "Matrix":
        """Standard matrix multiplication: self (m x n) * other (n x p) -> (m x p)."""
        if not isinstance(other, Matrix):
            raise MatrixError("Can only multiply by another Matrix.")
        if self._num_cols != other._num_rows:
            raise MatrixError(
                "Incompatible shapes for multiplication: "
                "{}x{} * {}x{}.".format(
                    self._num_rows, self._num_cols,
                    other._num_rows, other._num_cols,
                )
            )

        result = []
        for i in range(self._num_rows):
            new_row = []
            for j in range(other._num_cols):
                total = 0
                for k in range(self._num_cols):
                    total += self._rows[i][k] * other._rows[k][j]
                new_row.append(total)
            result.append(new_row)
        return Matrix(result)

    def transpose(self) -> "Matrix":
        result = []
        for j in range(self._num_cols):
            result.append([self._rows[i][j] for i in range(self._num_rows)])
        return Matrix(result)

    def _require_same_shape(self, other: "Matrix", op_name: str) -> None:
        if not isinstance(other, Matrix):
            raise MatrixError("Can only perform {} with another Matrix.".format(op_name))
        if self.shape != other.shape:
            raise MatrixError(
                "Matrices must have the same shape for {}: {} vs {}.".format(
                    op_name, self.shape, other.shape
                )
            )

    def __mul__(self, other: "Matrix | int | float") -> "Matrix":
        if isinstance(other, Matrix):
            return self.multiply(other)
        return self.scalar_multiply(other)

    def __rmul__(self, other: "int | float") -> "Matrix":
        return self.scalar_multiply(other)

    def __add__(self, other: "Matrix") -> "Matrix":
        return self.add(other)

    def __sub__(self, other: "Matrix") -> "Matrix":
        return self.subtract(other)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Matrix):
            return NotImplemented
        return self._rows == other._rows

    def to_list(self) -> "list[list[int | float]]":
        return [list(row) for row in self._rows]

    def __repr__(self) -> str:
        return "Matrix({})".format(self._rows)

    def __str__(self) -> str:
        widths = [0] * self._num_cols
        for i in range(self._num_rows):
            for j in range(self._num_cols):
                widths[j] = max(widths[j], len(str(self._rows[i][j])))

        lines = []
        for i in range(self._num_rows):
            cells = [
                str(self._rows[i][j]).rjust(widths[j]) for j in range(self._num_cols)
            ]
            lines.append("[ " + "  ".join(cells) + " ]")
        return "\n".join(lines)
