import numpy as np
import matplotlib.pyplot as plt
from typing import List, Any
from collections.abc import Callable

class Activation:
    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def dsigmoid(x: np.ndarray, grad_out: np.ndarray) -> np.ndarray:
        s = Activation.sigmoid(x)
        dz_dx = s * (1 - s)
        return dz_dx * grad_out

    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    @staticmethod
    def drelu(x: np.ndarray, grad_out: np.ndarray) -> np.ndarray:
        return (x > 0) * grad_out

class Loss:
    @staticmethod
    def mse(y_pred: np.ndarray, y_true: np.ndarray) -> np.floating:
        return np.mean((y_pred - y_true) ** 2)
    
    @staticmethod
    def dmse(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        return 2 * (y_pred - y_true) / y_pred.shape[0]
    
class Linear:
    def __init__(self, in_features: int, out_features: int) -> None:
        # how many input features, how many output (neurons)
        self.W: np.ndarray = np.random.randn(in_features, out_features) / np.sqrt(in_features) #xavier scaling initialisation
        self.b: np.ndarray = np.zeros(out_features)

        self.x: np.ndarray = np.zeros(0)
        self.dl_db = np.zeros(0)
        self.dl_dw = np.zeros(0)

    
    def forward(self, x: np.ndarray) -> np.ndarray:
        # forward pass! x can be batch matrix
        z = x @ self.W
        z += self.b
        self.x = x
        return z

    def backward(self, grad_out: np.ndarray):
        self.dl_dw = self.x.T @ grad_out
        self.dl_db = grad_out.sum(axis=0)
        dL_dx = grad_out @ self.W.T

        return dL_dx

class SGD:
    def __init__(self, layers: List[Linear], lr: float):
        self.layers = layers
        self.lr = lr
    
    def step(self):
        for layer in self.layers:
            layer.W -= self.lr * layer.dl_dw
            layer.b -= self.lr * layer.dl_db

class MLP:
    def __init__(self, layers: List[tuple[Linear, Callable | None, Callable | None]]): # activation function, d_activation
        self.layers = layers
        self.pre_activations: List[np.ndarray] = []
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.pre_activations = [] # reset each forward pass

        x_curr = x
        for layer in self.layers:
            z = layer[0].forward(x_curr)

            self.pre_activations.append(z)

            x_curr = layer[1](z) if layer[1] is not None else z
        
        return x_curr
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        grad_cur = grad
        for i, layer in reversed(list(enumerate(self.layers))):
            z = layer[2](self.pre_activations[i], grad_cur) if layer[2] is not None else grad_cur
            grad_cur = layer[0].backward(z)

        return grad_cur

if __name__ == "__main__":
    X = np.linspace(-2*np.pi, 2*np.pi, 1000).reshape(-1, 1)
    y = np.sin(X)

    layers = [(Linear(1, 16), Activation.relu, Activation.drelu), (Linear(16, 8), Activation.relu, Activation.drelu), (Linear(8, 1), None, None)]
    mlp = MLP(layers)
    sgd = SGD([layers[0][0], layers[1][0], layers[1][0]], 0.01)

    y_pred = mlp.forward(X)

    plt.ion()  # interactive mode on
    fig, ax = plt.subplots()
    ax.plot(X, y, label='true')
    pred_line, = ax.plot(X, y_pred, label='predicted')
    ax.legend()

    for i in range(5000):
        y_pred = mlp.forward(X)
        loss = Loss.mse(y_pred, y)
        print(loss)

        grad = Loss.dmse(y_pred, y)
        backward = mlp.backward(grad)

        y_pred = mlp.forward(X)
        pred_line.set_ydata(y_pred)
        plt.pause(0.001)

        sgd.step()
