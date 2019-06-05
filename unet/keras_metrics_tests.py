import unittest
import numpy as np
import keras_metrics
import keras.backend as K


class TestMetrics(unittest.TestCase):

    def setUp(self):
        self.true = np.array([[1., 0, 0], [0, 1, 0], [0, 0, 1]])
        self.pred = np.array([[1., 0, 0], [1, 0, 0], [0, 0, 1]])

    def test_tp(self):
        self.assertEqual(K.eval(keras_metrics.tp(self.true, self.pred)), 2)

    def test_fp(self):
        self.assertEqual(K.eval(keras_metrics.fp(self.true, self.pred)), 1)

    def test_tn(self):
        self.assertEqual(K.eval(keras_metrics.tn(self.true, self.pred)), 5)

    def test_fn(self):
        self.assertEqual(K.eval(keras_metrics.fn(self.true, self.pred)), 1)

    def test_precision(self):
        self.assertAlmostEqual(K.eval(keras_metrics.precision(self.true, self.pred)), 2./3.)

    def test_recall(self):
        self.assertAlmostEqual(K.eval(keras_metrics.recall(self.true, self.pred)), 2./3.)

    def test_f_score(self):
        self.assertAlmostEqual(K.eval(keras_metrics.f_score(self.true, self.pred)), 2./3.)

    def test_pixel_error(self):
        self.assertEqual(K.eval(keras_metrics.pixel_error(self.true, self.pred)), 1.0)

    def test_rand_error(self):
        self.assertEqual(K.eval(keras_metrics.rand_error(self.true, self.pred)), 1.0)

    def test_wrapping_error(self):
        self.assertEqual(K.eval(keras_metrics.wrapping_error(self.true, self.pred)), 1.0)

    def test_dice(self):
        self.assertEqual(K.eval(keras_metrics.dice(self.true, self.pred)).tolist(), 2./3.)

    def test_iou(self):
        self.assertEqual(K.eval(keras_metrics.iou(self.true, self.pred)).tolist(), 2./3.)


if __name__ == '__main__':
    unittest.main()
