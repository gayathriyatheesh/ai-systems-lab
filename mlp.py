import numpy as np


def init_params():
    w1 = np.random.randn(2, 4) * 0.01
    b1 = np.zeros((1, 4))

    w2 = np.random.rand(4, 1) * 0.01
    b2 = np.zeros((1,1))
    return {
        "w1": w1,
        "b1": b1,
        "w2": w2,
        "b2": b2
    }
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def forward(X, params):
    w1 = params["w1"]
    b1 = params["b1"]
    w2 = params["w2"]
    b2 = params["b2"]

    z1 = X @ w1 + b1
    a1 = sigmoid(z1)

    z2 = a1 @ w2 + b2

    cache = {
        "X": X,
        "z1": z1,
        "a1": a1,
        "z2": z2
    }

    return z2, cache

def loss(z2, y):
    error = z2 - y
    return 0.5 * np.mean(error ** 2)

def backward(cache, y, params):
    X = cache["X"]
    a1 = cache["a1"]
    z2 = cache["z2"]
    w2 = params["w2"]
    delta2 = z2 - y
    N = X.shape[0]
    dw2 = (a1.T @ delta2) / N
    db2 = np.mean(delta2, axis=0, keepdims=True)
    delta1 = (delta2 @ w2.T) * a1 * (1 - a1)
    dw1 = X.T @ delta1 / N
    db1 = np.mean(delta1, axis=0, keepdims=True)
    grads = {
        "w1" : dw1,
        "b1" : db1,
        "w2" : dw2,
        "b2" : db2
    }
    return grads
def step(params, grads, lr):
    params["w1"] -= lr * grads["w1"]
    params["b1"] -= lr * grads["b1"]
    params["w2"] -= lr * grads["w2"]
    params["b2"] -= lr * grads["b2"]

    return params
if __name__ == "__main__":
    np.random.seed(42)
    N = 100
    X = np.random.rand(N, 2)
    y = (2 * X[:, 0:1] - 3 * X[:, 1:2] + 1)
    epochs = 1000
    lr = 0.1
    params = init_params()

    for epoch in range(epochs):
        z2, cache = forward(X, params)
        l = loss(z2, y)
        grads = backward(cache, y, params)
        params = step(params, grads, lr)

        if epoch % 100 == 0:
           print(f"Epoch {epoch}, Loss: {l:.6f}")


    z2, cache = forward(X, params)
    grads = backward(cache, y, params)
    epsilon = 1e-5

    original = params["w2"][1, 0]
    params["w2"][1, 0] = original + epsilon
    z2_plus, _ =forward(X, params)
    loss_plus = loss(z2_plus, y)

    params["w2"][1, 0] = original - epsilon
    z2_minus, _ = forward(X, params)
    loss_minus = loss(z2_minus, y)

    numerical_grad = (loss_plus - loss_minus) / (2 * epsilon)
    params["w2"][1, 0] = original

    analytical_grad = grads["w2"][1, 0]
    difference = abs(numerical_grad - analytical_grad)
    print(f"Numerical gradient: {numerical_grad}")
    print(f"Analytical gradient: {analytical_grad}")
    print(f"Difference: {difference}")

    original = params["w1"][0, 0]
    params["w1"][0, 0] = original + epsilon
    z2_plus, _ =forward(X, params)
    loss_plus = loss(z2_plus, y)

    params["w1"][0, 0] = original - epsilon
    z2_minus, _ = forward(X, params)
    loss_minus = loss(z2_minus, y)

    numerical_grad = (loss_plus - loss_minus) / (2 * epsilon)
    params["w1"][0, 0] = original

    analytical_grad = grads["w1"][0, 0]
    difference = abs(numerical_grad - analytical_grad)
    print(f"Numerical gradient: {numerical_grad}")
    print(f"Analytical gradient: {analytical_grad}")
    print(f"Difference: {difference}")

    original = params["b2"][0, 0]
    params["b2"][0, 0] = original + epsilon
    z2_plus, _ =forward(X, params)
    loss_plus = loss(z2_plus, y)

    params["b2"][0, 0] = original - epsilon
    z2_minus, _ = forward(X, params)
    loss_minus = loss(z2_minus, y)

    numerical_grad = (loss_plus - loss_minus) / (2 * epsilon)
    params["b2"][0, 0] = original

    analytical_grad = grads["b2"][0]
    difference = abs(numerical_grad - analytical_grad)
    print(f"Numerical gradient: {numerical_grad}")
    print(f"Analytical gradient: {analytical_grad}")
    print(f"Difference: {difference}")

    original = params["b1"][0, 0]
    params["b1"][0, 0] = original + epsilon
    z2_plus, _ =forward(X, params)
    loss_plus = loss(z2_plus, y)

    params["b1"][0, 0] = original - epsilon
    z2_minus, _ = forward(X, params)
    loss_minus = loss(z2_minus, y)

    numerical_grad = (loss_plus - loss_minus) / (2 * epsilon)
    params["b1"][0, 0] = original

    analytical_grad = grads["b1"][0]
    difference = abs(numerical_grad - analytical_grad)
    print(f"Numerical gradient: {numerical_grad}")
    print(f"Analytical gradient: {analytical_grad}")
    print(f"Difference: {difference}")

    N = 20
    X_test = np.random.rand(N, 2)
    Y_test =  2 * X_test[:, 0:1] - 3 * X_test[:, 1:2] + 1
    predictions, _ = forward(X_test, params)
    loss_test = loss(predictions, Y_test)
    print("Test Loss:", loss_test)