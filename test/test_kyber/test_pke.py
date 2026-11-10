import unittest

from cryptography.hazmat.primitives.asymmetric.mlkem import MLKEM768PrivateKey

from kyber.pke import KyberPKE


class TestKyberPKE(unittest.TestCase):

    def test_random_values(self):
        # Using parameters from ML-KEM-512.
        m = b"\x00\x12\x34\x56" * 8
        kyber = KyberPKE.from_seed(
            n=256,
            q=3329,
            k=2,
            nu1=3,
            nu2=2,
            du=10,
            dv=4,
            seed=b"\x00" * 32,
        )
        c = kyber.encrypt(m, r=b"\x00" * 32)
        m_decrypted = kyber.decrypt(c)

        self.assertEqual(32, len(m_decrypted))
        self.assertEqual(m, m_decrypted)

    def test_key_gen(self):
        seed = b"\x01" * 32

        kyber = KyberPKE.from_seed(
            n=256,
            q=3329,
            k=3,
            nu1=2,
            nu2=2,
            du=10,
            dv=4,
            seed=seed,
        )
        public_key_actual = kyber.ek
        A_ntt_actual, t_ntt_actual = kyber.deserialize_encryption_key()

        private_key_expected = MLKEM768PrivateKey.from_seed_bytes(seed + seed)
        public_key_expected = private_key_expected.public_key().public_bytes_raw()

        # Load raw bytes into their numerical representation for easier debugging.
        kyber = KyberPKE(
            n=256,
            q=3329,
            k=3,
            nu1=2,
            nu2=2,
            du=10,
            dv=4,
            ek=public_key_expected,
            dk=b"\x00" * (384 * 3),
        )
        A_ntt_expected, t_ntt_expected = kyber.deserialize_encryption_key()

        self.assertEqual(A_ntt_actual, A_ntt_expected)
        self.assertEqual(t_ntt_actual, t_ntt_expected)

        self.assertEqual(len(public_key_expected), len(public_key_actual))
        self.assertEqual(public_key_expected, public_key_actual)
