import numpy as np
from typing import List

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
        self.W = np.random.randn(in_features, out_features) / np.sqrt(in_features) #xavier scaling initialisation
        self.b = np.zeros(out_features)

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
    def __init__(self, layers: List[Linear], lr: np.floating):
        self.layers = layers
        self.lr = lr
    
    def step(self):
        for layer in self.layers:
            layer.W -= self.lr * layer.dl_dw
            layer.b -= self.lr * layer.dl_db

class MLP:
    def __init__(self, layers: List[tuple]):
        pass

if __name__ == "__main__":
    X = np.linspace(-2*np.pi, 2*np.pi, 1000).reshape(-1, 1)
    y = np.sin(X)