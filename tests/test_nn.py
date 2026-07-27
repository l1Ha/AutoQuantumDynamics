import unittest
import numpy as np
from autoquantum.nn.model import FeedForwardNN


class TestNNModel(unittest.TestCase):
    def test_forward_shape(self):
        model = FeedForwardNN([1, 10, 1], activation="tanh", seed=42)
        x = np.linspace(0, 1, 50)
        y = model.predict(x)
        self.assertEqual(y.shape, (50,))

    def test_training_converges(self):
        model = FeedForwardNN([1, 20, 20, 1], activation="tanh", seed=42)
        x = np.linspace(0, 1, 100).reshape(-1, 1)
        y = (x ** 2).flatten()
        for _ in range(100):
            model.train_step(x, y, lr=0.01)
        pred = model.predict(x)
        loss = np.mean((pred - y) ** 2)
        self.assertLess(loss, 0.1)


if __name__ == "__main__":
    unittest.main()
