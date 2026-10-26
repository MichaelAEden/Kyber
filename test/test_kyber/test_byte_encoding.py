import unittest

import numpy as np

from kyber.byte_encoding import byte_encode, byte_decode


class TestByteEncoding(unittest.TestCase):

    def test_encode_8_bit_integers(self):
        q = 3329
        d = 8

        data = np.array([1, 1, 1, 1, 1, 1, 1, 255])
        data_encoded_expected = b"\x01\x01\x01\x01\x01\x01\x01\xff"
        data_encoded_actual = byte_encode(data, d=d, q=q)

        self.assertEqual(data_encoded_expected, data_encoded_actual)

    def test_encode_12_bit_integers(self):
        q = 3329
        d = 12

        data = np.array([1, 1, 1, 1, 1, 1, 1, 3328])
        data_encoded_expected = b"\x01\x10\x00\x01\x10\x00\x01\x10\x00\x01\x00\xd0"
        data_encoded_actual = byte_encode(data, d=d, q=q)

        self.assertEqual(data_encoded_expected, data_encoded_actual)

    def test_encode_8_bit_data_out_of_range(self):
        q = 3329
        d = 8

        data = np.array([1, 1, 1, 1, 1, 1, 1, 256])
        with self.assertRaises(ValueError):
            byte_encode(data, d=d, q=q)

    def test_encode_12_bit_data_out_of_range(self):
        q = 3329
        d = 12

        data = np.array([1, 1, 1, 1, 1, 1, 1, 3329])
        with self.assertRaises(ValueError):
            byte_encode(data, d=d, q=q)


class TestByteDecoding(unittest.TestCase):

    def test_decode_8_bit_integers(self):
        q = 3329
        d = 8

        data = b"\x01\x01\x01\x01\x01\x01\x01\xff"
        data_decoded_expected = np.array([1, 1, 1, 1, 1, 1, 1, 255])
        data_decoded_actual = byte_decode(data, d=d, q=q)

        np.testing.assert_array_equal(data_decoded_actual, data_decoded_expected)

    def test_decode_12_bit_integers(self):
        q = 3329
        d = 12

        data = b"\x01\x10\x00\x01\x10\x00\x01\x10\x00\x01\x00\xd0"
        data_decoded_expected = np.array([1, 1, 1, 1, 1, 1, 1, 3328])
        data_decoded_actual = byte_decode(data, d=d, q=q)

        np.testing.assert_array_equal(data_decoded_expected, data_decoded_actual)

    def test_decode_12_bit_data_out_of_range(self):
        q = 3329
        d = 12

        data = b"\x01\x10\x00\x01\x10\x00\x01\x10\x00\x01\x10\xd0"
        with self.assertRaises(ValueError):
            byte_decode(data, d=d, q=q)
