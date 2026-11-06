import numpy as np

from mlp import backward, forward, init_params, loss


def test_gradient_check():
    np.random.seed(0)
    X = np.random.randn(5, 2)
    y = np.random.randn(5, 1)
    params = init_params()
    z2, cache = forward(X, params)
    grads = backward(cache, y, params)
    eps = 1e-5

    for name in ["w1", "b1", "w2", "b2"]:
        p = params[name]
        g = grads[name]
        for idx in np.ndindex(*p.shape):
            old = p[idx]
            p[idx] = old + eps
            z2, _ = forward(X, params)
            loss_plus = loss(z2, y)
            p[idx] = old - eps
            z2, _ = forward(X, params)
            loss_minus = loss(z2, y)
            numerical_grad = (loss_plus - loss_minus) / (2 * eps)
            p[idx] = old
            rel = abs(g[idx] - numerical_grad) / (abs(g[idx]) + abs(numerical_grad) + 1e-12)
            
            assert rel < 1e-6, (name, idx, rel)