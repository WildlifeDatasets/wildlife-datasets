import unittest
import numpy as np
from wildlife_datasets import evaluation

tol = 0.000001

y_true1 = [1,1,2,3,3]
y_pred1 = [1,1,2,3,1]

encoder1 = {1: 'a', 2: 'b', 3: 'c'}
encoder2 = {1: 'new', 2: 2, 3: 3}
encoder3 = {1: 'a', 2: 'b', 3: 0}

def encode(y, encoder):
    return [encoder[x] for x in y]

class TestEvaluation(unittest.TestCase):   
    def test_accuracy(self):
        expected_value = 4/5
        metric = evaluation.accuracy

        y_true = y_true1
        y_pred = y_pred1
        self.assertAlmostEqual(metric(y_true, y_pred), expected_value, delta=tol)

        y_true = encode(y_true1, encoder1)
        y_pred = encode(y_pred1, encoder1)
        self.assertAlmostEqual(metric(y_true, y_pred), expected_value, delta=tol)

        y_true = encode(y_true1, encoder2)
        y_pred = encode(y_pred1, encoder2)
        self.assertRaises(Exception, metric, y_true, y_pred)        
        self.assertAlmostEqual(metric(y_true, y_pred, unknown_class='new'), expected_value, delta=tol)

        y_true = encode(y_true1, encoder3)
        y_pred = encode(y_pred1, encoder3)
        self.assertRaises(Exception, metric, y_true, y_pred)        
        self.assertAlmostEqual(metric(y_true, y_pred, unknown_class=0), expected_value, delta=tol)

    def test_balanced_accuracy(self):
        expected_value = 5/6
        metric = evaluation.balanced_accuracy

        y_true = y_true1
        y_pred = y_pred1
        self.assertAlmostEqual(metric(y_true, y_pred), expected_value, delta=tol)

        y_true = encode(y_true1, encoder1)
        y_pred = encode(y_pred1, encoder1)
        self.assertAlmostEqual(metric(y_true, y_pred), expected_value, delta=tol)

        y_true = encode(y_true1, encoder2)
        y_pred = encode(y_pred1, encoder2)
        self.assertRaises(Exception, metric, y_true, y_pred)        
        self.assertAlmostEqual(metric(y_true, y_pred, unknown_class='new'), expected_value, delta=tol)

        y_true = encode(y_true1, encoder3)
        y_pred = encode(y_pred1, encoder3)
        self.assertRaises(Exception, metric, y_true, y_pred)        
        self.assertAlmostEqual(metric(y_true, y_pred, unknown_class=0), expected_value, delta=tol)

    def test_class_average_accuracy(self):
        expected_value = 13/15
        metric = evaluation.class_average_accuracy

        y_true = y_true1
        y_pred = y_pred1
        self.assertAlmostEqual(metric(y_true, y_pred), expected_value, delta=tol)

        y_true = encode(y_true1, encoder1)
        y_pred = encode(y_pred1, encoder1)
        self.assertAlmostEqual(metric(y_true, y_pred), expected_value, delta=tol)

        y_true = encode(y_true1, encoder2)
        y_pred = encode(y_pred1, encoder2)
        self.assertRaises(Exception, metric, y_true, y_pred)        
        self.assertAlmostEqual(metric(y_true, y_pred, unknown_class='new'), expected_value, delta=tol)

        y_true = encode(y_true1, encoder3)
        y_pred = encode(y_pred1, encoder3)
        self.assertRaises(Exception, metric, y_true, y_pred)        
        self.assertAlmostEqual(metric(y_true, y_pred, unknown_class=0), expected_value, delta=tol)

# TODO: finish

if __name__ == '__main__':
    unittest.main()

