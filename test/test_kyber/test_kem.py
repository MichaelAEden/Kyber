import unittest

from kyber.kem import KyberKEM


class TestKyberKEM(unittest.TestCase):

    def test_random_values(self):
        # Using parameters from ML-KEM-512.
        kyber = KyberKEM(
            n=256,
            q=3329,
            k=2,
            nu1=3,
            nu2=2,
            seed=b"\x01" * 32,
        )
        K, c = kyber.encapsulate(seed=2027)
        K_prime = kyber.decapsulate(c)

        self.assertEqual(K, K_prime)
