import unittest

import numpy as np

from kyber.byte_encoding import bit_encode, bit_decode, bits_to_bytes, bytes_to_bits


class TestBitEncode(unittest.TestCase):

    def test_encode_8_bit_integers(self):
        d = 8

        integers = np.array([0, 1, 255])
        # fmt: off
        bits_expected = np.array([
            0, 0, 0, 0, 0, 0, 0, 0,
            1, 0, 0, 0, 0, 0, 0, 0,
            1, 1, 1, 1, 1, 1, 1, 1,
        ], dtype=bool)
        # fmt: on
        bits_actual = bit_encode(integers, d=d)

        np.testing.assert_array_equal(bits_expected, bits_actual)

    def test_encode_12_bit_integers(self):
        d = 12

        integers = np.array([0, 1, 4095])
        # fmt: off
        bits_expected = np.array([
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        ], dtype=bool)
        # fmt: on
        bits_actual = bit_encode(integers, d=d)

        np.testing.assert_array_equal(bits_expected, bits_actual)

    def test_encode_out_of_range(self):
        d = 12

        integers = np.array([1, 4096])
        with self.assertRaises(OverflowError):
            bit_encode(integers, d=d)


class TestBitDecode(unittest.TestCase):

    def test_decode_8_bit_integers(self):
        d = 8

        # fmt: off
        bits = np.array([
            0, 0, 0, 0, 0, 0, 0, 0,
            1, 0, 0, 0, 0, 0, 0, 0,
            1, 1, 1, 1, 1, 1, 1, 1,
        ], dtype=bool)
        # fmt: on
        integers_expected = np.array([0, 1, 255])
        integers_actual = bit_decode(bits, d=d)

        np.testing.assert_array_equal(integers_expected, integers_actual)

    def test_decode_12_bit_integers(self):
        d = 12

        # fmt: off
        bits = np.array([
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        ], dtype=bool)
        # fmt: on
        integers_expected = np.array([0, 1, 4095])
        integers_actual = bit_decode(bits, d=d)

        np.testing.assert_array_equal(integers_expected, integers_actual)

    def test_invalid_input_length(self):
        d = 12

        # fmt: off
        bits = np.array([
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        ], dtype=bool)
        # fmt: on

        with self.assertRaises(ValueError):
            bit_decode(bits, d=d)


class TestBitToBytes(unittest.TestCase):

    def test(self):
        # fmt: off
        bits = np.array([
            0, 0, 0, 0, 0, 0, 0, 0,
            1, 0, 0, 0, 0, 0, 0, 0,
            1, 1, 1, 1, 1, 1, 1, 1,
        ], dtype=bool)
        # fmt: on
        bytes_expected = b"\x00\x01\xff"
        bytes_actual = bits_to_bytes(bits)

        self.assertEqual(bytes_expected, bytes_actual)

    def test_invalid_input_length(self):
        # fmt: off
        bits = np.array([
            0, 0, 0, 0, 0, 0, 0, 0,
            1, 0, 0, 0, 0, 0, 0, 0,
            1, 1, 1, 1, 1, 1,
        ], dtype=bool)
        # fmt: on

        with self.assertRaises(ValueError):
            bits_to_bytes(bits)


class TestBytesToBits(unittest.TestCase):

    def test(self):
        # fmt: off
        bits_expected = np.array([
            0, 0, 0, 0, 0, 0, 0, 0,
            1, 0, 0, 0, 0, 0, 0, 0,
            1, 1, 1, 1, 1, 1, 1, 1,
        ], dtype=bool)
        # fmt: on
        bits_actual = bytes_to_bits(b"\x00\x01\xff")

        np.testing.assert_array_equal(bits_expected, bits_actual)
