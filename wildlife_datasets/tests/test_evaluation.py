import unittest
from collections.abc import Iterable

import numpy as np

from wildlife_datasets import metrics

tol = 0.000001

y_true1 = [1,1,2,3,3]
y_pred1 = [1,1,2,3,1]

y_true2 = [1,1,2,3,3]
y_pred2 = [1,1,2,3,4]

y_true_rank1 = 1
y_pred_rank1 = [1, 2, 2]

y_true_rank2 = 2
y_pred_rank2 = [1, 2, 2]

encoder1 = {1: 'a', 2: 'b', 3: 'c', 4: 'd'}
encoder2 = {1: 'new', 2: 2, 3: 3, 4: 4}
encoder3 = {1: 0, 2: 'b', 3: 'c', 4: 'd'}

def encode(y, encoder):
    if isinstance(y, Iterable):
        return [encoder[x] for x in y]
    else:
        return encoder[y]

def macro_f1(ps, rs):
    return np.mean([0 if (p,r)==(0,0) else 2*p*r/(p+r) for (p, r) in zip(ps, rs)])

class TestEvaluation(unittest.TestCase):   
    def test_accuracy1(self):
        expected_value = 4/5
        metric = metrics.accuracy
        y_true_basis = y_true1
        y_pred_basis = y_pred1

        y_true = y_true_basis
        y_pred = y_pred_basis
        self.assertAlmostEqual(metric(y_true, y_pred), expected_value, delta=tol)

        y_true = encode(y_true_basis, encoder1)
        y_pred = encode(y_pred_basis, encoder1)
        self.assertAlmostEqual(metric(y_true, y_pred), expected_value, delta=tol)

        y_true = encode(y_true_basis, encoder2)
        y_pred = encode(y_pred_basis, encoder2)
        self.assertRaises(Exception, metric, y_true, y_pred)        
        self.assertAlmostEqual(metric(y_true, y_pred, new_class='new'), expected_value, delta=tol)

        y_true = encode(y_true_basis, encoder3)
        y_pred = encode(y_pred_basis, encoder3)
        self.assertRaises(Exception, metric, y_true, y_pred)        
        self.assertAlmostEqual(metric(y_true, y_pred, new_class=0), expected_value, delta=tol)

    def test_accuracy2(self):
        expected_value = 4/5
        metric = metrics.accuracy
        y_true_basis = y_true2
        y_pred_basis = y_pred2

        y_true = y_true_basis
        y_pred = y_pred_basis
        self.assertAlmostEqual(metric(y_true, y_pred), expected_value, delta=tol)

        y_true = encode(y_true_basis, encoder1)
        y_pred = encode(y_pred_basis, encoder1)
        self.assertAlmostEqual(metric(y_true, y_pred), expected_value, delta=tol)

        y_true = encode(y_true_basis, encoder2)
        y_pred = encode(y_pred_basis, encoder2)
        self.assertRaises(Exception, metric, y_true, y_pred)        
        self.assertAlmostEqual(metric(y_true, y_pred, new_class='new'), expected_value, delta=tol)

        y_true = encode(y_true_basis, encoder3)
        y_pred = encode(y_pred_basis, encoder3)
        self.assertRaises(Exception, metric, y_true, y_pred)        
        self.assertAlmostEqual(metric(y_true, y_pred, new_class=0), expected_value, delta=tol)

    def test_balanced_accuracy1(self):
        expected_value = 5/6
        metric = metrics.balanced_accuracy
        y_true_basis = y_true1
        y_pred_basis = y_pred1

        y_true = y_true_basis
        y_pred = y_pred_basis
        self.assertAlmostEqual(metric(y_true, y_pred), expected_value, delta=tol)

        y_true = encode(y_true_basis, encoder1)
        y_pred = encode(y_pred_basis, encoder1)
        self.assertAlmostEqual(metric(y_true, y_pred), expected_value, delta=tol)

        y_true = encode(y_true_basis, encoder2)
        y_pred = encode(y_pred_basis, encoder2)
        self.assertRaises(Exception, metric, y_true, y_pred)        
        self.assertAlmostEqual(metric(y_true, y_pred, new_class='new'), expected_value, delta=tol)

        y_true = encode(y_true_basis, encoder3)
        y_pred = encode(y_pred_basis, encoder3)
        self.assertRaises(Exception, metric, y_true, y_pred)        
        self.assertAlmostEqual(metric(y_true, y_pred, new_class=0), expected_value, delta=tol)

    def test_balanced_accuracy2(self):
        expected_value = 5/6
        metric = metrics.balanced_accuracy
        y_true_basis = y_true2
        y_pred_basis = y_pred2

        y_true = y_true_basis
        y_pred = y_pred_basis
        self.assertAlmostEqual(metric(y_true, y_pred), expected_value, delta=tol)

        y_true = encode(y_true_basis, encoder1)
        y_pred = encode(y_pred_basis, encoder1)
        self.assertAlmostEqual(metric(y_true, y_pred), expected_value, delta=tol)

        y_true = encode(y_true_basis, encoder2)
        y_pred = encode(y_pred_basis, encoder2)
        self.assertRaises(Exception, metric, y_true, y_pred)        
        self.assertAlmostEqual(metric(y_true, y_pred, new_class='new'), expected_value, delta=tol)

        y_true = encode(y_true_basis, encoder3)
        y_pred = encode(y_pred_basis, encoder3)
        self.assertRaises(Exception, metric, y_true, y_pred)        
        self.assertAlmostEqual(metric(y_true, y_pred, new_class=0), expected_value, delta=tol)

    def test_class_average_accuracy1(self):
        expected_value = 13/15
        metric = metrics.class_average_accuracy
        y_true_basis = y_true1
        y_pred_basis = y_pred1

        y_true = y_true_basis
        y_pred = y_pred_basis
        self.assertAlmostEqual(metric(y_true, y_pred), expected_value, delta=tol)

        y_true = encode(y_true_basis, encoder1)
        y_pred = encode(y_pred_basis, encoder1)
        self.assertAlmostEqual(metric(y_true, y_pred), expected_value, delta=tol)

        y_true = encode(y_true_basis, encoder2)
        y_pred = encode(y_pred1, encoder2)
        self.assertRaises(Exception, metric, y_true, y_pred)        
        self.assertAlmostEqual(metric(y_true, y_pred, new_class='new'), expected_value, delta=tol)

        y_true = encode(y_true_basis, encoder3)
        y_pred = encode(y_pred_basis, encoder3)
        self.assertRaises(Exception, metric, y_true, y_pred)        
        self.assertAlmostEqual(metric(y_true, y_pred, new_class=0), expected_value, delta=tol)

    def test_class_average_accuracy2(self):
        expected_value = 9/10
        metric = metrics.class_average_accuracy
        y_true_basis = y_true2
        y_pred_basis = y_pred2

        y_true = y_true_basis
        y_pred = y_pred_basis
        self.assertAlmostEqual(metric(y_true, y_pred), expected_value, delta=tol)

        y_true = encode(y_true_basis, encoder1)
        y_pred = encode(y_pred_basis, encoder1)
        self.assertAlmostEqual(metric(y_true, y_pred), expected_value, delta=tol)

        y_true = encode(y_true_basis, encoder2)
        y_pred = encode(y_pred_basis, encoder2)
        self.assertRaises(Exception, metric, y_true, y_pred)        
        self.assertAlmostEqual(metric(y_true, y_pred, new_class='new'), expected_value, delta=tol)

        y_true = encode(y_true_basis, encoder3)
        y_pred = encode(y_pred_basis, encoder3)
        self.assertRaises(Exception, metric, y_true, y_pred)        
        self.assertAlmostEqual(metric(y_true, y_pred, new_class=0), expected_value, delta=tol)

    def test_precision1(self):
        expected_value = 8/9
        metric = metrics.precision
        y_true_basis = y_true1
        y_pred_basis = y_pred1

        y_true = y_true_basis
        y_pred = y_pred_basis
        self.assertAlmostEqual(metric(y_true, y_pred), expected_value, delta=tol)

        y_true = encode(y_true_basis, encoder1)
        y_pred = encode(y_pred_basis, encoder1)
        self.assertAlmostEqual(metric(y_true, y_pred), expected_value, delta=tol)

        y_true = encode(y_true_basis, encoder2)
        y_pred = encode(y_pred_basis, encoder2)
        self.assertRaises(Exception, metric, y_true, y_pred)
        self.assertAlmostEqual(metric(y_true, y_pred, new_class='new'), expected_value, delta=tol)

        y_true = encode(y_true_basis, encoder3)
        y_pred = encode(y_pred_basis, encoder3)
        self.assertRaises(Exception, metric, y_true, y_pred)
        self.assertAlmostEqual(metric(y_true, y_pred, new_class=0), expected_value, delta=tol)

    def test_precision2(self):
        expected_value = 3/4
        metric = metrics.precision
        y_true_basis = y_true2
        y_pred_basis = y_pred2
        
        y_true = y_true_basis
        y_pred = y_pred_basis
        self.assertAlmostEqual(metric(y_true, y_pred), expected_value, delta=tol)
        
        y_true = encode(y_true_basis, encoder1)
        y_pred = encode(y_pred_basis, encoder1)
        self.assertAlmostEqual(metric(y_true, y_pred), expected_value, delta=tol)

        y_true = encode(y_true_basis, encoder2)
        y_pred = encode(y_pred_basis, encoder2)
        self.assertRaises(Exception, metric, y_true, y_pred)
        self.assertAlmostEqual(metric(y_true, y_pred, new_class='new'), expected_value, delta=tol)

        y_true = encode(y_true_basis, encoder3)
        y_pred = encode(y_pred_basis, encoder3)
        self.assertRaises(Exception, metric, y_true, y_pred)
        self.assertAlmostEqual(metric(y_true, y_pred, new_class=0), expected_value, delta=tol)

    def test_recall1(self):
        expected_value = 5/6
        metric = metrics.recall
        y_true_basis = y_true1
        y_pred_basis = y_pred1

        y_true = y_true_basis
        y_pred = y_pred_basis
        self.assertAlmostEqual(metric(y_true, y_pred), expected_value, delta=tol)
        self.assertAlmostEqual(metric(y_true, y_pred, ignore_empty=True), expected_value, delta=tol)

        y_true = encode(y_true_basis, encoder1)
        y_pred = encode(y_pred_basis, encoder1)
        self.assertAlmostEqual(metric(y_true, y_pred), expected_value, delta=tol)
        self.assertAlmostEqual(metric(y_true, y_pred, ignore_empty=True), expected_value, delta=tol)

        y_true = encode(y_true_basis, encoder2)
        y_pred = encode(y_pred_basis, encoder2)
        self.assertRaises(Exception, metric, y_true, y_pred)
        self.assertRaises(Exception, metric, y_true, y_pred, ignore_empty=True)
        self.assertAlmostEqual(metric(y_true, y_pred, new_class='new'), expected_value, delta=tol)
        self.assertAlmostEqual(metric(y_true, y_pred, new_class='new', ignore_empty=True), expected_value, delta=tol)

        y_true = encode(y_true_basis, encoder3)
        y_pred = encode(y_pred_basis, encoder3)
        self.assertRaises(Exception, metric, y_true, y_pred)
        self.assertRaises(Exception, metric, y_true, y_pred, ignore_empty=True)
        self.assertAlmostEqual(metric(y_true, y_pred, new_class=0), expected_value, delta=tol)
        self.assertAlmostEqual(metric(y_true, y_pred, new_class=0, ignore_empty=True), expected_value, delta=tol)

    def test_recall2(self):
        expected_value = 5/8
        expected_value_mod = 5/6
        metric = metrics.recall
        y_true_basis = y_true2
        y_pred_basis = y_pred2

        y_true = y_true_basis
        y_pred = y_pred_basis
        self.assertAlmostEqual(metric(y_true, y_pred), expected_value, delta=tol)
        self.assertAlmostEqual(metric(y_true, y_pred, ignore_empty=True), expected_value_mod, delta=tol)

        y_true = encode(y_true_basis, encoder1)
        y_pred = encode(y_pred_basis, encoder1)
        self.assertAlmostEqual(metric(y_true, y_pred), expected_value, delta=tol)
        self.assertAlmostEqual(metric(y_true, y_pred, ignore_empty=True), expected_value_mod, delta=tol)

        y_true = encode(y_true_basis, encoder2)
        y_pred = encode(y_pred_basis, encoder2)
        self.assertRaises(Exception, metric, y_true, y_pred)
        self.assertRaises(Exception, metric, y_true, y_pred, ignore_empty=True)
        self.assertAlmostEqual(metric(y_true, y_pred, new_class='new'), expected_value, delta=tol)
        self.assertAlmostEqual(metric(y_true, y_pred, new_class='new', ignore_empty=True), expected_value_mod, delta=tol)

        y_true = encode(y_true_basis, encoder3)
        y_pred = encode(y_pred_basis, encoder3)
        self.assertRaises(Exception, metric, y_true, y_pred)
        self.assertRaises(Exception, metric, y_true, y_pred, ignore_empty=True)
        self.assertAlmostEqual(metric(y_true, y_pred, new_class=0), expected_value, delta=tol)
        self.assertAlmostEqual(metric(y_true, y_pred, new_class=0, ignore_empty=True), expected_value_mod, delta=tol)

    def test_f11(self):
        expected_value = macro_f1([2/3,1/1,1/1], [2/2,1/1,1/2])
        metric = metrics.f1
        y_true_basis = y_true1
        y_pred_basis = y_pred1

        y_true = y_true_basis
        y_pred = y_pred_basis
        self.assertAlmostEqual(metric(y_true, y_pred), expected_value, delta=tol)

        y_true = encode(y_true_basis, encoder1)
        y_pred = encode(y_pred_basis, encoder1)
        self.assertAlmostEqual(metric(y_true, y_pred), expected_value, delta=tol)

        y_true = encode(y_true_basis, encoder2)
        y_pred = encode(y_pred_basis, encoder2)
        self.assertRaises(Exception, metric, y_true, y_pred)
        self.assertAlmostEqual(metric(y_true, y_pred, new_class='new'), expected_value, delta=tol)

        y_true = encode(y_true_basis, encoder3)
        y_pred = encode(y_pred_basis, encoder3)
        self.assertRaises(Exception, metric, y_true, y_pred)
        self.assertAlmostEqual(metric(y_true, y_pred, new_class=0), expected_value, delta=tol)

    def test_f12(self):
        expected_value = macro_f1([2/2,1/1,1/1,0/1], [2/2,1/1,1/2,0])
        metric = metrics.f1
        y_true_basis = y_true2
        y_pred_basis = y_pred2

        y_true = y_true_basis
        y_pred = y_pred_basis
        self.assertAlmostEqual(metric(y_true, y_pred), expected_value, delta=tol)

        y_true = encode(y_true_basis, encoder1)
        y_pred = encode(y_pred_basis, encoder1)
        self.assertAlmostEqual(metric(y_true, y_pred), expected_value, delta=tol)

        y_true = encode(y_true_basis, encoder2)
        y_pred = encode(y_pred_basis, encoder2)
        self.assertRaises(Exception, metric, y_true, y_pred)
        self.assertAlmostEqual(metric(y_true, y_pred, new_class='new'), expected_value, delta=tol)

        y_true = encode(y_true_basis, encoder3)
        y_pred = encode(y_pred_basis, encoder3)
        self.assertRaises(Exception, metric, y_true, y_pred)
        self.assertAlmostEqual(metric(y_true, y_pred, new_class=0), expected_value, delta=tol)

    def test_normalized_accuracy1(self):
        expected_value1 = 2/3
        expected_value2 = 1
        metric = metrics.normalized_accuracy
        y_true_basis = y_true1
        y_pred_basis = y_pred1

        y_true = y_true_basis
        y_pred = y_pred_basis
        self.assertRaises(Exception, metric, y_true, y_pred)
        self.assertRaises(Exception, metric, y_true, y_pred, 1)
        for mu in np.arange(0, 1, step=0.2):
            self.assertAlmostEqual(metric(y_true, y_pred, 1, mu), mu*expected_value1+(1-mu)*expected_value2, delta=tol)
        
        y_true = encode(y_true_basis, encoder1)
        y_pred = encode(y_pred_basis, encoder1)
        self.assertRaises(Exception, metric, y_true, y_pred)
        self.assertRaises(Exception, metric, y_true, y_pred, 1)
        for mu in np.arange(0, 1, step=0.2):
            self.assertAlmostEqual(metric(y_true, y_pred, 'a', mu), mu*expected_value1+(1-mu)*expected_value2, delta=tol)

    def test_average_precision1(self):
        expected_value = 1
        metric = metrics.average_precision
        y_true_basis = y_true_rank1
        y_pred_basis = y_pred_rank1

        y_true = y_true_basis
        y_pred = y_pred_basis        
        self.assertAlmostEqual(metric(y_true, y_pred), expected_value, delta=tol)
        
        y_true = encode(y_true_basis, encoder1)
        y_pred = encode(y_pred_basis, encoder1)
        self.assertAlmostEqual(metric(y_true, y_pred), expected_value, delta=tol)

    def test_average_precision2(self):
        expected_value = 7/12
        metric = metrics.average_precision
        y_true_basis = y_true_rank2
        y_pred_basis = y_pred_rank2

        y_true = y_true_basis
        y_pred = y_pred_basis        
        self.assertAlmostEqual(metric(y_true, y_pred), expected_value, delta=tol)
        
        y_true = encode(y_true_basis, encoder1)
        y_pred = encode(y_pred_basis, encoder1)
        self.assertAlmostEqual(metric(y_true, y_pred), expected_value, delta=tol)

    def test_mean_average_precision1(self):
        expected_value = (1+7/12)/2
        metric = metrics.mean_average_precision
        y_true_basis = [y_true_rank1, y_true_rank2]
        y_pred_basis = [y_pred_rank1, y_pred_rank2]

        y_true = y_true_basis
        y_pred = y_pred_basis        
        self.assertAlmostEqual(metric(y_true, y_pred), expected_value, delta=tol)

    def test_baks(self):
        identity_test_only = [[], [4], [1], [1, 2], [1, 3], [1, 2, 3]]
        expected_values = [5/6, 5/6, 3/4, 1/2, 1, np.nan]
        metric = metrics.BAKS
        y_true_basis = y_true1
        y_pred_basis = y_pred1

        for id_test_only, expected_value in zip(identity_test_only, expected_values):
            y_true = y_true_basis
            y_pred = y_pred_basis
            if id_test_only == [1, 2, 3]:
                np.testing.assert_equal(metric(y_true, y_pred, id_test_only), expected_value)
            else:
                self.assertAlmostEqual(metric(y_true, y_pred, id_test_only), expected_value, delta=tol)
                self.assertAlmostEqual(metric(np.array(y_true), y_pred, id_test_only), expected_value, delta=tol)
                self.assertAlmostEqual(metric(y_true, np.array(y_pred), id_test_only), expected_value, delta=tol)
                self.assertAlmostEqual(metric(y_true, y_pred, np.array(id_test_only)), expected_value, delta=tol)
         
        for id_test_only, expected_value in zip(identity_test_only, expected_values):
            y_true = encode(y_true_basis, encoder1)
            y_pred = encode(y_pred_basis, encoder1)
            if id_test_only == [1, 2, 3]:
                id_test_only = encode(id_test_only, encoder1)                
                np.testing.assert_equal(metric(y_true, y_pred, id_test_only), expected_value)
            else:
                id_test_only = encode(id_test_only, encoder1)
                self.assertAlmostEqual(metric(y_true, y_pred, id_test_only), expected_value, delta=tol)
        
        for id_test_only, expected_value in zip(identity_test_only, expected_values):
            # The case [1, 3] is actually ok because the mixed labels are removed
            y_true = encode(y_true_basis, encoder2)
            y_pred = encode(y_pred_basis, encoder2)
            if id_test_only == [1, 2, 3]:
                id_test_only = encode(id_test_only, encoder2)
                np.testing.assert_equal(metric(y_true, y_pred, id_test_only), expected_value)
            elif id_test_only != [1, 3]:
                id_test_only = encode(id_test_only, encoder2)
                self.assertRaises(Exception, metric, y_true, y_pred, id_test_only)
         
        for id_test_only, expected_value in zip(identity_test_only, expected_values):
            # The last case is actually ok because the mixed labels are removed
            y_true = encode(y_true_basis, encoder3)
            y_pred = encode(y_pred_basis, encoder3)
            if id_test_only == [1, 2, 3]:
                id_test_only = encode(id_test_only, encoder3)
                np.testing.assert_equal(metric(y_true, y_pred, id_test_only), expected_value)
            elif id_test_only != [1, 3]:
                id_test_only = encode(id_test_only, encoder3)
                self.assertRaises(Exception, metric, y_true, y_pred, id_test_only)

    def test_baus(self):
        identity_test_only = [[], [4], [1], [1, 2], [1, 3], [1, 2, 3]]
        expected_values = [np.nan, np.nan, 1, 1/2, 3/4, 1/2]
        new_class_basis = 1        
        metric = metrics.BAUS
        y_true_basis = y_true1
        y_pred_basis = y_pred1

        for id_test_only, expected_value in zip(identity_test_only, expected_values):
            y_true = y_true_basis
            y_pred = y_pred_basis
            new_class = new_class_basis
            if id_test_only == [] or id_test_only == [4]:
                np.testing.assert_equal(metric(y_true, y_pred, id_test_only, new_class), expected_value)
            else:
                self.assertAlmostEqual(metric(y_true, y_pred, id_test_only, new_class), expected_value, delta=tol)
                self.assertAlmostEqual(metric(np.array(y_true), y_pred, id_test_only, new_class), expected_value, delta=tol)
                self.assertAlmostEqual(metric(y_true, np.array(y_pred), id_test_only, new_class), expected_value, delta=tol)
                self.assertAlmostEqual(metric(y_true, y_pred, np.array(id_test_only), new_class), expected_value, delta=tol)
        
        for id_test_only, expected_value in zip(identity_test_only, expected_values):
            y_true = encode(y_true_basis, encoder1)
            y_pred = encode(y_pred_basis, encoder1)
            new_class = encode(new_class_basis, encoder1)
            if id_test_only == [] or id_test_only == [4]:
                id_test_only = encode(id_test_only, encoder1)                
                np.testing.assert_equal(metric(y_true, y_pred, id_test_only, new_class), expected_value)
            else:
                id_test_only = encode(id_test_only, encoder1)
                self.assertAlmostEqual(metric(y_true, y_pred, id_test_only, new_class), expected_value, delta=tol)
         
if __name__ == '__main__':
    unittest.main()

