import numpy as np
from api import Linear, Mlp

if __name__ == "__main__":
    rows = 1000
    X = np.linspace(0, 2 * np.pi, rows, dtype=np.float64).reshape(rows, 1)
    y = np.sin(X)

    l1 = Linear(1, 32, "TANH")
    l2 = Linear(32, 32, "TANH")
    l3 = Linear(32, 32, "TANH")
    l4 = Linear(32, 1, None)

    mlp = Mlp([l1, l2, l3, l4])
    mlp.mlp_run(X, y, rows, 1, 100001, 0.001)