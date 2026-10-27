import unittest

from kyber.compression import compress, decompress


class TestCompress(unittest.TestCase):

    def test(self):
        # See Example 6.9 in the following document:
        # https://cryptography101.ca/wp-content/uploads/lattice-based-cryptography.pdf
        self.assertEqual(746, compress(x=2426, q=3329, d=10))


class TestDecompress(unittest.TestCase):

    def test(self):
        # See Example 6.9 in the following document:
        # https://cryptography101.ca/wp-content/uploads/lattice-based-cryptography.pdf
        self.assertEqual(2425, decompress(y=746, q=3329, d=10))
