import unittest

import numpy as np

from wildlife_datasets import splits

tol = 0.1

class TestLcg(unittest.TestCase):   
    def test_lcg1(self):
        n = 10
        n_rep = 1000

        lcg = splits.Lcg(0)
        permutation_sum = np.zeros(n)        
        for _ in range(n_rep):
            permutation_sum += lcg.random_permutation(n)

        expected_value = n_rep*(n-1)/2
        self.assertAlmostEqual(np.min(permutation_sum), expected_value, delta=expected_value*tol)
        self.assertAlmostEqual(np.max(permutation_sum), expected_value, delta=expected_value*tol)        

    def test_lcg2(self):
        n = 10
        n_rep = 1000
        tol = 0.05

        lcg = splits.Lcg(0)        
        permutation_sum = np.zeros(n)        
        for _ in range(n_rep):
            permutation_sum += lcg.random_shuffle(list(range(n)))

        expected_value = n_rep*(n-1)/2
        self.assertAlmostEqual(np.min(permutation_sum), expected_value, delta=expected_value*tol)
        self.assertAlmostEqual(np.max(permutation_sum), expected_value, delta=expected_value*tol)        


if __name__ == '__main__':
    unittest.main()

