import unittest

import numpy as np

from math_.matrix import Matrix


class TestMatrix(unittest.TestCase):

    def test_iter(self):
        a = Matrix[int](
            (
                (1, 2, 3),
                (4, 5, 6),
            )
        )
        values_expected = [1, 2, 3, 4, 5, 6]
        values_actual = list(iter(a))
        self.assertListEqual(values_expected, values_actual)

    def test_equal(self):
        a = Matrix[int](
            (
                (1, 2, 3),
                (4, 5, 6),
            )
        )
        b = Matrix[int](
            (
                (1, 2, 3),
                (4, 5, 6),
            )
        )
        self.assertEqual(a, b)

    def test_not_equal_values(self):
        a = Matrix[int](
            (
                (1, 2, 3),
                (4, 5, 6),
            )
        )
        b = Matrix[int](
            (
                (1, 2, 3),
                (4, 5, 0),
            )
        )
        self.assertNotEqual(a, b)

    def test_transpose(self):
        a = Matrix[int](
            (
                (1, 2, 3),
                (4, 5, 6),
            )
        )
        b = a.transpose
        self.assertEqual(b.shape, (3, 2))
        self.assertEqual(b[0, 1], 4)
        self.assertEqual(b[1, 0], 2)

    def test_not_equal_sizes(self):
        a = Matrix[int](
            (
                (1, 2, 3),
                (4, 5, 6),
                (7, 8, 9),
            )
        )
        b = Matrix[int](
            (
                (1, 2, 3),
                (4, 5, 6),
            )
        )
        self.assertNotEqual(a, b)

    def test_multiply_integer_matrix(self):
        a = Matrix[int](
            (
                (1, 2, 3),
                (4, 5, 6),
            )
        )
        b = Matrix[int](
            (
                (1, 2, 3),
                (4, 5, 6),
                (7, 8, 9),
            )
        )
        matrix_expected = Matrix[int](((30, 36, 42), (66, 81, 96)))
        matrix_actual = a * b
        np.testing.assert_array_equal(matrix_expected, matrix_actual)
