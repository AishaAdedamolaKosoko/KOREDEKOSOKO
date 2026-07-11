import unittest

from cnn_defect_model import compute_classification_metrics


class EvaluationMetricsTests(unittest.TestCase):
    def test_binary_metrics(self):
        metrics = compute_classification_metrics(
            y_true=[0, 1, 1, 0],
            y_pred=[0, 1, 0, 0],
            labels=[0, 1],
            positive_label=1,
        )

        self.assertAlmostEqual(metrics["accuracy"], 0.75, places=4)
        self.assertAlmostEqual(metrics["precision"], 1.0, places=4)
        self.assertAlmostEqual(metrics["recall"], 0.5, places=4)
        self.assertAlmostEqual(metrics["specificity"], 1.0, places=4)
        self.assertAlmostEqual(metrics["f1_score"], 2 / 3, places=4)
        self.assertAlmostEqual(metrics["error_rate"], 0.25, places=4)


if __name__ == "__main__":
    unittest.main()
