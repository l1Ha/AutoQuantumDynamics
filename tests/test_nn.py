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

    def test_linear_output_layer(self):
        # Regression: the output layer must be linear. With a tanh (or other
        # bounded) output activation, predictions saturate within [-1, 1] and
        # cannot represent PES values outside that range. Here the network is
        # driven to a constant pre-activation of 3.0, which a squashed output
        # layer would clip to ~1.0.
        model = FeedForwardNN([1, 4, 1], activation="tanh", seed=0)
        for w in model.weights[:-1]:
            w[:] = 0.0
        for b in model.biases[:-1]:
            b[:] = 0.0
        model.weights[-1][:] = 1.0
        model.biases[-1][:] = 3.0

        x = np.linspace(-2, 2, 21)
        pred = model.predict(x)
        np.testing.assert_allclose(pred, 3.0, atol=1e-12)

    def test_fit_outside_tanh_range(self):
        # Regression: targets beyond [-1, 1] must be fittable now that the
        # output layer is linear. The prediction-mean assertion would fail
        # unconditionally with a tanh output layer (bounded by 1.0).
        model = FeedForwardNN([1, 32, 1], activation="tanh", seed=42)
        x = np.linspace(0, 1, 60).reshape(-1, 1)
        y = (3.0 * x + 1.0).flatten()  # range [1, 4]
        for _ in range(500):
            model.train_step(x, y, lr=0.01)
        pred = model.predict(x)
        self.assertGreater(np.mean(pred), 1.5)
        self.assertLess(np.mean((pred - y) ** 2), 0.05)

    def test_finite_difference_mse_gradient(self):
        # The backprop gradients returned by _backward must match the numerical
        # gradient of the mean squared error up to the factor 1/2 implied by
        # the delta = (pred - y) output convention
        # (d(MSE)/dW = 2 * delta-based dw for a linear output layer).
        rng = np.random.RandomState(7)
        model = FeedForwardNN([1, 6, 1], activation="tanh", seed=3)
        x = rng.uniform(-1, 1, size=(12, 1))
        y = rng.uniform(-1, 1, size=12)

        activations, zs = model.forward(x)
        dw, db = model._backward(x, y, activations, zs)

        def mse_loss():
            return np.mean((model.predict(x) - y) ** 2)

        eps = 1e-6
        for i in range(len(model.weights)):
            for arr, grad in ((model.weights[i], dw[i]), (model.biases[i], db[i])):
                num_grad = np.zeros_like(arr)
                it = np.nditer(arr, flags=["multi_index"])
                while not it.finished:
                    idx = it.multi_index
                    orig = arr[idx]
                    arr[idx] = orig + eps
                    loss_plus = mse_loss()
                    arr[idx] = orig - eps
                    loss_minus = mse_loss()
                    arr[idx] = orig
                    num_grad[idx] = (loss_plus - loss_minus) / (2 * eps)
                    it.iternext()
                np.testing.assert_allclose(
                    grad, 0.5 * num_grad, rtol=1e-5, atol=1e-9
                )


if __name__ == "__main__":
    unittest.main()
