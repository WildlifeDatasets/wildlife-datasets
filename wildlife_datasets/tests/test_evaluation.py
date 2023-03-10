import unittest
import numpy as np
from wildlife_datasets import evaluation

tol = 0.000001

y_true1 = [1,1,2,3,3]
y_pred1 = [1,1,2,3,1]

y_true2 = [1,1,2,3,3]
y_pred2 = [1,1,2,3,4]

encoder1 = {1: 'a', 2: 'b', 3: 'c', 4: 'd'}
encoder2 = {1: 'new', 2: 2, 3: 3, 4: 4}
encoder3 = {1: 0, 2: 'b', 3: 'c', 4: 'd'}

def encode(y, encoder):
    return [encoder[x] for x in y]

def macro_f1(ps, rs):
    return np.mean([0 if (p,r)==(0,0) else 2*p*r/(p+r) for (p, r) in zip(ps, rs)])

class TestEvaluation(unittest.TestCase):   
    def test_accuracy1(self):
        expected_value = 4/5
        metric = evaluation.accuracy
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
        self.assertAlmostEqual(metric(y_true, y_pred, unknown_class='new'), expected_value, delta=tol)

        y_true = encode(y_true_basis, encoder3)
        y_pred = encode(y_pred_basis, encoder3)
        self.assertRaises(Exception, metric, y_true, y_pred)        
        self.assertAlmostEqual(metric(y_true, y_pred, unknown_class=0), expected_value, delta=tol)

    def test_accuracy2(self):
        expected_value = 4/5
        metric = evaluation.accuracy
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
        self.assertAlmostEqual(metric(y_true, y_pred, unknown_class='new'), expected_value, delta=tol)

        y_true = encode(y_true_basis, encoder3)
        y_pred = encode(y_pred_basis, encoder3)
        self.assertRaises(Exception, metric, y_true, y_pred)        
        self.assertAlmostEqual(metric(y_true, y_pred, unknown_class=0), expected_value, delta=tol)

    def test_balanced_accuracy1(self):
        expected_value = 5/6
        metric = evaluation.balanced_accuracy
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
        self.assertAlmostEqual(metric(y_true, y_pred, unknown_class='new'), expected_value, delta=tol)

        y_true = encode(y_true_basis, encoder3)
        y_pred = encode(y_pred_basis, encoder3)
        self.assertRaises(Exception, metric, y_true, y_pred)        
        self.assertAlmostEqual(metric(y_true, y_pred, unknown_class=0), expected_value, delta=tol)

    def test_balanced_accuracy2(self):
        expected_value = 5/6
        metric = evaluation.balanced_accuracy
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
        self.assertAlmostEqual(metric(y_true, y_pred, unknown_class='new'), expected_value, delta=tol)

        y_true = encode(y_true_basis, encoder3)
        y_pred = encode(y_pred_basis, encoder3)
        self.assertRaises(Exception, metric, y_true, y_pred)        
        self.assertAlmostEqual(metric(y_true, y_pred, unknown_class=0), expected_value, delta=tol)

    def test_class_average_accuracy1(self):
        expected_value = 13/15
        metric = evaluation.class_average_accuracy
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
        self.assertAlmostEqual(metric(y_true, y_pred, unknown_class='new'), expected_value, delta=tol)

        y_true = encode(y_true_basis, encoder3)
        y_pred = encode(y_pred_basis, encoder3)
        self.assertRaises(Exception, metric, y_true, y_pred)        
        self.assertAlmostEqual(metric(y_true, y_pred, unknown_class=0), expected_value, delta=tol)

    def test_class_average_accuracy2(self):
        expected_value = 9/10
        metric = evaluation.class_average_accuracy
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
        self.assertAlmostEqual(metric(y_true, y_pred, unknown_class='new'), expected_value, delta=tol)

        y_true = encode(y_true_basis, encoder3)
        y_pred = encode(y_pred_basis, encoder3)
        self.assertRaises(Exception, metric, y_true, y_pred)        
        self.assertAlmostEqual(metric(y_true, y_pred, unknown_class=0), expected_value, delta=tol)

    def test_precision1(self):
        expected_value = 8/9
        metric = evaluation.precision
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
        self.assertAlmostEqual(metric(y_true, y_pred, unknown_class='new'), expected_value, delta=tol)

        y_true = encode(y_true_basis, encoder3)
        y_pred = encode(y_pred_basis, encoder3)
        self.assertRaises(Exception, metric, y_true, y_pred)
        self.assertAlmostEqual(metric(y_true, y_pred, unknown_class=0), expected_value, delta=tol)

    def test_precision2(self):
        expected_value = 3/4
        metric = evaluation.precision
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
        self.assertAlmostEqual(metric(y_true, y_pred, unknown_class='new'), expected_value, delta=tol)

        y_true = encode(y_true_basis, encoder3)
        y_pred = encode(y_pred_basis, encoder3)
        self.assertRaises(Exception, metric, y_true, y_pred)
        self.assertAlmostEqual(metric(y_true, y_pred, unknown_class=0), expected_value, delta=tol)

    def test_recall1(self):
        expected_value = 5/6
        metric = evaluation.recall
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
        self.assertAlmostEqual(metric(y_true, y_pred, unknown_class='new'), expected_value, delta=tol)
        self.assertAlmostEqual(metric(y_true, y_pred, unknown_class='new', ignore_empty=True), expected_value, delta=tol)

        y_true = encode(y_true_basis, encoder3)
        y_pred = encode(y_pred_basis, encoder3)
        self.assertRaises(Exception, metric, y_true, y_pred)
        self.assertRaises(Exception, metric, y_true, y_pred, ignore_empty=True)
        self.assertAlmostEqual(metric(y_true, y_pred, unknown_class=0), expected_value, delta=tol)
        self.assertAlmostEqual(metric(y_true, y_pred, unknown_class=0, ignore_empty=True), expected_value, delta=tol)

    def test_recall2(self):
        expected_value = 5/8
        expected_value_mod = 5/6
        metric = evaluation.recall
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
        self.assertAlmostEqual(metric(y_true, y_pred, unknown_class='new'), expected_value, delta=tol)
        self.assertAlmostEqual(metric(y_true, y_pred, unknown_class='new', ignore_empty=True), expected_value_mod, delta=tol)

        y_true = encode(y_true_basis, encoder3)
        y_pred = encode(y_pred_basis, encoder3)
        self.assertRaises(Exception, metric, y_true, y_pred)
        self.assertRaises(Exception, metric, y_true, y_pred, ignore_empty=True)
        self.assertAlmostEqual(metric(y_true, y_pred, unknown_class=0), expected_value, delta=tol)
        self.assertAlmostEqual(metric(y_true, y_pred, unknown_class=0, ignore_empty=True), expected_value_mod, delta=tol)

    def test_f11(self):
        expected_value = macro_f1([2/3,1/1,1/1], [2/2,1/1,1/2])
        metric = evaluation.f1
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
        self.assertAlmostEqual(metric(y_true, y_pred, unknown_class='new'), expected_value, delta=tol)

        y_true = encode(y_true_basis, encoder3)
        y_pred = encode(y_pred_basis, encoder3)
        self.assertRaises(Exception, metric, y_true, y_pred)
        self.assertAlmostEqual(metric(y_true, y_pred, unknown_class=0), expected_value, delta=tol)

    def test_f12(self):
        expected_value = macro_f1([2/2,1/1,1/1,0/1], [2/2,1/1,1/2,0])
        metric = evaluation.f1
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
        self.assertAlmostEqual(metric(y_true, y_pred, unknown_class='new'), expected_value, delta=tol)

        y_true = encode(y_true_basis, encoder3)
        y_pred = encode(y_pred_basis, encoder3)
        self.assertRaises(Exception, metric, y_true, y_pred)
        self.assertAlmostEqual(metric(y_true, y_pred, unknown_class=0), expected_value, delta=tol)

    def test_normalized_accuracy1(self):
        expected_value1 = 1
        expected_value2 = 2/3
        metric = evaluation.normalized_accuracy
        y_true_basis = y_true1
        y_pred_basis = y_pred1

        y_true = y_true_basis
        y_pred = y_pred_basis
        self.assertRaises(Exception, metric, y_true, y_pred)
        self.assertRaises(Exception, metric, y_true, y_pred, 1)
        self.assertRaises(Exception, metric, y_true, y_pred, -1, 1)
        for mu in np.arange(0, 1, step=0.2):
            self.assertAlmostEqual(metric(y_true, y_pred, 1, mu), mu*expected_value1+(1-mu)*expected_value2, delta=tol)
        
        y_true = encode(y_true_basis, encoder1)
        y_pred = encode(y_pred_basis, encoder1)
        self.assertRaises(Exception, metric, y_true, y_pred)
        self.assertRaises(Exception, metric, y_true, y_pred, 1)
        self.assertRaises(Exception, metric, y_true, y_pred, -1, 1)
        for mu in np.arange(0, 1, step=0.2):
            self.assertAlmostEqual(metric(y_true, y_pred, 'a', mu), mu*expected_value1+(1-mu)*expected_value2, delta=tol)


if __name__ == '__main__':
    unittest.main()

