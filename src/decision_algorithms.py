import numpy as np
from _typing import *

def entropy(Y: ArrayLike) -> float:
    res = 0
    unique_values = np.unique(Y)
    for value in unique_values:
        Y_value = [y for y in Y if y == value]
        p = len(Y_value) / len(Y)
        res += p * np.log2(p)
        
    return -res

def information_gain(col_values: ArrayLike, Y: ArrayLike) -> float:
    pre_entropy = entropy(Y)
    unique_values = np.unique(col_values)
    post_entropy = 0
    for value in unique_values:
        Y_filtered = [y for i, y in enumerate(Y) if col_values[i] == value]
        post_entropy += (len(Y_filtered) / len(Y)) * entropy(Y_filtered)
    return pre_entropy - post_entropy
    

def id3(X: MatrixLike, Y: ArrayLike):
    
    
def c4_5(X: MatrixLike, Y: ArrayLike):
    raise NotImplementedError