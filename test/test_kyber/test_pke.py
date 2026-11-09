import unittest

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
