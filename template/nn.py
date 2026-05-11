import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    z = 1 / (1 + np.exp(-x))
    return z

def relu(x: np.ndarray) -> np.ndarray:
    z = np.maximum(0, x)
    return z

class Linear:
    def __init__(self, in_features: int, out_features: int) -> None:
        # how many input features, how many output (neurons)
        self.W = np.random.randn(in_features, out_features) / np.sqrt(in_features) #xavier scaling initialisation
        self.b = np.zeros(out_features)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        # forward pass! x can be batch matrix
        z = x @ self.W
        z += self.b
        return z



if __name__ == "__main__":
    print("hello world")