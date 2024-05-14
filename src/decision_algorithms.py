import numpy as np
from _typing import *

def entropy(col_values: ArrayLike, Y: ArrayLike) -> float:
    Y_bin = np.bincount(Y)
    res = 0
    for i in Y_bin:
        p = i / len(Y)
        res -= p * np.log2(p)
        
    return res

def id3(X: MatrixLike, Y: ArrayLike):
    
    
def c4_5(X: MatrixLike, Y: ArrayLike):
    raise NotImplementedError