from ctypes import *
import numpy as np
from pathlib import Path
from typing import List

proj_root = Path(__file__).resolve().parent.parent
DLL_PATH = proj_root / "camel.dll"

c = CDLL(DLL_PATH)

c.api_linear_create.argtypes = [c_int, c_int, c_char_p]
c.api_linear_create.restype = c_void_p

c.api_mlp_create.argtypes = [POINTER(c_void_p), c_int]
c.api_mlp_create.restype = c_void_p

c.api_mlp_run.argtypes = [c_void_p, POINTER(c_double), POINTER(c_double), c_int, c_int, c_int, c_double]
c.api_mlp_run.restype = None

class Linear:
    def __init__(self, in_features: int, out_features: int, activation: str | None) -> None:
        self.in_features = in_features
        self.out_features = out_features
        
        if (activation not in ("RELU", "SIGMOID", "TANH", None)):
            raise ValueError("incorrect activation bruh")
        
        if activation == None:
            self.activation = None
        else:
            self.activation = activation.encode('utf-8')

        self.layer_addr: c_void_p = c.api_linear_create(self.in_features, self.out_features, self.activation)

class Mlp:
    def __init__(self, layers: List[Linear]) -> None:
        self.layers = layers
        self.n_layers = len(layers)

        self.c_layer_arr = (c_void_p * self.n_layers)(*[l.layer_addr for l in layers]) # void*[n_layers]

        self.mlp_addr: c_void_p = c.api_mlp_create(self.c_layer_arr, self.n_layers)
    
    def mlp_run(self, X: np.ndarray, y: np.ndarray, rows: int, cols: int, epochs: int, lr: float):
        X = np.ascontiguousarray(X, np.float64)
        y = np.ascontiguousarray(y, np.float64)

        c.api_mlp_run(self.mlp_addr, X.ctypes.data_as(POINTER(c_double)), y.ctypes.data_as(POINTER(c_double)), rows, cols, epochs, lr)