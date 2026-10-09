from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np


class Activation(ABC):
    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def backward(self, x: np.ndarray, grad_out: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class Sigmoid(Activation):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def backward(self, x: np.ndarray, grad_out: np.ndarray) -> np.ndarray:
        output = self.forward(x)
        return output * (1 - output) * grad_out


class ReLU(Activation):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def backward(self, x: np.ndarray, grad_out: np.ndarray) -> np.ndarray:
        return (x > 0) * grad_out


class Identity(Activation):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return x

    def backward(self, x: np.ndarray, grad_out: np.ndarray) -> np.ndarray:
        return grad_out


class Loss:
    @staticmethod
    def mse(y_pred: np.ndarray, y_true: np.ndarray) -> np.floating:
        return np.mean((y_pred - y_true) ** 2)

    @staticmethod
    def dmse(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        return 2 * (y_pred - y_true) / y_pred.shape[0]


class Linear:
    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: Activation | None = None,
    ) -> None:
        self.W: np.ndarray = (
            np.random.randn(in_features, out_features) / np.sqrt(in_features)
        )
        self.b: np.ndarray = np.zeros(out_features)
        self.activation = activation if activation is not None else Identity()

        self.x: np.ndarray = np.zeros(0)
        self.z: np.ndarray = np.zeros(0)
        self.dl_db = np.zeros(0)
        self.dl_dw = np.zeros(0)

    def forward(self, x: np.ndarray) -> np.ndarray:
        # forward pass! x can be batch matrix
        z = x @ self.W
        z += self.b
        self.x = x
        self.z = z
        return self.activation.forward(z)

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        grad_out = self.activation.backward(self.z, grad_out)
        self.dl_dw = self.x.T @ grad_out
        self.dl_db = grad_out.sum(axis=0)
        return grad_out @ self.W.T


class SGD:
    def __init__(self, layers: Sequence[Linear], lr: float) -> None:
        self.layers = list(layers)
        self.lr = lr

    def step(self) -> None:
        for layer in self.layers:
            layer.W -= self.lr * layer.dl_dw
            layer.b -= self.lr * layer.dl_db


class MLP:
    def __init__(self, layers: Sequence[Linear]) -> None:
        self.layers = list(layers)

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad: np.ndarray) -> np.ndarray:
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    X = np.linspace(-2*np.pi, 2*np.pi, 1000).reshape(-1, 1)
    y = np.sin(X)

    layers = [Linear(1, 16, ReLU()), Linear(16, 8, ReLU()), Linear(8, 1)]
    mlp = MLP(layers)
    sgd = SGD(layers, 0.005)

    y_pred = mlp.forward(X)

    plt.ion()  # interactive mode on
    fig, ax = plt.subplots()
    ax.plot(X, y, label='true')
    pred_line, = ax.plot(X, y_pred, label='predicted')
    ax.legend()

    for i in range(10000):
        y_pred = mlp.forward(X)
        loss = Loss.mse(y_pred, y)
        print(loss)

        grad = Loss.dmse(y_pred, y)
        mlp.backward(grad)

        pred_line.set_ydata(y_pred)
        plt.pause(0.001)

        sgd.step()
