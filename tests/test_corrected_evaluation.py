import unittest

import numpy as np

from analysis.corrected_evaluation import average_precision, roc_auc


class MetricTests(unittest.TestCase):
    def test_perfect_auc(self):
        y = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        self.assertAlmostEqual(roc_auc(y, scores), 1.0)

    def test_reversed_auc(self):
        y = np.array([0, 0, 1, 1])
        scores = np.array([0.9, 0.8, 0.2, 0.1])
        self.assertAlmostEqual(roc_auc(y, scores), 0.0)

    def test_auc_ties(self):
        y = np.array([0, 1])
        scores = np.array([0.5, 0.5])
        self.assertAlmostEqual(roc_auc(y, scores), 0.5)

    def test_average_precision_perfect(self):
        y = np.array([0, 1, 0, 1])
        scores = np.array([0.1, 0.9, 0.2, 0.8])
        self.assertAlmostEqual(average_precision(y, scores), 1.0)

    def test_weighted_auc(self):
        y = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.9, 0.8, 1.0])
        unweighted = roc_auc(y, scores)
        weighted = roc_auc(y, scores, np.array([10.0, 1.0, 1.0, 1.0]))
        self.assertGreater(weighted, unweighted)


if __name__ == "__main__":
    unittest.main()
