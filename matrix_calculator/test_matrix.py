"""Unit tests for the Matrix class using the standard library unittest module."""

import unittest

from matrix import Matrix, MatrixError


class TestMatrix(unittest.TestCase):
    def test_multiply_basic(self):
        a = Matrix([[1, 2, 3], [4, 5, 6]])
        b = Matrix([[7, 8], [9, 10], [11, 12]])
        result = a.multiply(b)
        self.assertEqual(result.to_list(), [[58, 64], [139, 154]])

    def test_multiply_identity(self):
        a = Matrix([[1, 2], [3, 4]])
        identity = Matrix([[1, 0], [0, 1]])
        self.assertEqual(a.multiply(identity), a)

    def test_multiply_incompatible(self):
        a = Matrix([[1, 2, 3]])
        b = Matrix([[1, 2, 3]])
        with self.assertRaises(MatrixError):
            a.multiply(b)

    def test_operator_overload(self):
        a = Matrix([[1, 2], [3, 4]])
        b = Matrix([[5, 6], [7, 8]])
        self.assertEqual((a * b).to_list(), [[19, 22], [43, 50]])

    def test_add_and_subtract(self):
        a = Matrix([[1, 2], [3, 4]])
        b = Matrix([[5, 6], [7, 8]])
        self.assertEqual((a + b).to_list(), [[6, 8], [10, 12]])
        self.assertEqual((b - a).to_list(), [[4, 4], [4, 4]])

    def test_scalar_multiply(self):
        a = Matrix([[1, 2], [3, 4]])
        self.assertEqual(a.scalar_multiply(3).to_list(), [[3, 6], [9, 12]])
        self.assertEqual((2 * a).to_list(), [[2, 4], [6, 8]])

    def test_transpose(self):
        a = Matrix([[1, 2, 3], [4, 5, 6]])
        self.assertEqual(a.transpose().to_list(), [[1, 4], [2, 5], [3, 6]])

    def test_invalid_construction(self):
        with self.assertRaises(MatrixError):
            Matrix([])
        with self.assertRaises(MatrixError):
            Matrix([[1, 2], [3]])
        with self.assertRaises(MatrixError):
            Matrix([[1, True]])

    def test_shape_mismatch(self):
        a = Matrix([[1, 2]])
        b = Matrix([[1, 2], [3, 4]])
        with self.assertRaises(MatrixError):
            a.add(b)


if __name__ == "__main__":
    unittest.main()
