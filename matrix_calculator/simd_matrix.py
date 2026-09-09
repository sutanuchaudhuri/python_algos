"""SIMD matrix multiply that calls a compiled C kernel via ctypes (no NumPy).

This is the version that actually captures NumPy's real advantage: the hot loop
runs in native, SIMD-vectorized machine code instead of the Python interpreter.

  * Data lives in a flat, row-major ``array('d')`` (contiguous C doubles).
  * ``multiply`` hands the raw buffers to ``matmul`` in ``libmatmul`` (compiled
    from ``matmul.c`` with ``-O3 -march=native -ffast-math``), whose inner loop
    the compiler auto-vectorizes to SIMD (NEON on arm64, AVX on x86-64).

Build the shared library first with ``make libmatmul`` (or just ``make bench``).
"""

import ctypes
import os
from array import array


class SimdMatrixError(Exception):
    """Raised when a SimdMatrix operation cannot be performed."""


_LIB = None


def _lib_path() -> str:
    ext = ".dylib" if os.uname().sysname == "Darwin" else ".so"
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "libmatmul" + ext)


def _load_lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is None:
        path = _lib_path()
        if not os.path.exists(path):
            raise SimdMatrixError(
                "Native library not found at {}. Build it with `make libmatmul`.".format(path)
            )
        lib = ctypes.CDLL(path)
        c_dbl_p = ctypes.POINTER(ctypes.c_double)
        lib.matmul.argtypes = [c_dbl_p, c_dbl_p, c_dbl_p,
                               ctypes.c_int, ctypes.c_int, ctypes.c_int]
        lib.matmul.restype = None
        _LIB = lib
    return _LIB


def is_available() -> bool:
    """Return True if the compiled native library can be loaded."""
    try:
        _load_lib()
        return True
    except SimdMatrixError:
        return False


class SimdMatrix:
    """A dense matrix backed by a flat contiguous ``array('d')`` buffer."""

    __slots__ = ("rows", "cols", "_data")

    def __init__(self, data: "list[list[float]]") -> None:
        if not data or not data[0]:
            raise SimdMatrixError("matrix must be a non-empty list of rows.")
        cols = len(data[0])
        flat = array("d")
        for row in data:
            if len(row) != cols:
                raise SimdMatrixError("All rows must have the same number of columns.")
            flat.extend(row)
        self.rows = len(data)
        self.cols = cols
        self._data = flat

    @classmethod
    def _from_flat(cls, rows: int, cols: int, flat: array) -> "SimdMatrix":
        obj = cls.__new__(cls)
        obj.rows = rows
        obj.cols = cols
        obj._data = flat
        return obj

    @property
    def shape(self) -> "tuple[int, int]":
        return (self.rows, self.cols)

    def get(self, r: int, c: int) -> float:
        return self._data[r * self.cols + c]

    def multiply(self, other: "SimdMatrix") -> "SimdMatrix":
        """Multiply via the compiled SIMD C kernel."""
        if self.cols != other.rows:
            raise SimdMatrixError(
                "Incompatible shapes for multiplication: "
                "{}x{} * {}x{}.".format(self.rows, self.cols, other.rows, other.cols)
            )

        lib = _load_lib()
        m, n, p = self.rows, self.cols, other.cols
        result = array("d", bytes(8 * m * p))

        # Zero-copy views of the contiguous buffers as C double arrays.
        a_buf = (ctypes.c_double * len(self._data)).from_buffer(self._data)
        b_buf = (ctypes.c_double * len(other._data)).from_buffer(other._data)
        c_buf = (ctypes.c_double * len(result)).from_buffer(result)

        lib.matmul(a_buf, b_buf, c_buf, m, n, p)
        return SimdMatrix._from_flat(m, p, result)

    def to_list(self) -> "list[list[float]]":
        cols = self.cols
        return [list(self._data[i * cols:(i + 1) * cols]) for i in range(self.rows)]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SimdMatrix):
            return NotImplemented
        return self.shape == other.shape and self._data == other._data

    def __repr__(self) -> str:
        return "SimdMatrix(rows={}, cols={})".format(self.rows, self.cols)
