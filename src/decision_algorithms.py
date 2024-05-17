import numpy as np
from _typing import *

def entropy(Y: ArrayLike) -> float:
    '''
    Calculates the entropy of a given target array.
    
    Parameters
    ----------
    Y: ArrayLike 
        Target column to calculate the entropy of.
        
    Returns
    -------
    entropy: float
        Entropy value of the given column.
    '''
    res = 0
    unique_values = np.unique(Y)
    for value in unique_values:
        Y_value = [y for y in Y if y == value]
        p = len(Y_value) / len(Y)
        res += p * np.log2(p)
        
    return -res

def information_gain(col_values: ArrayLike, Y: ArrayLike) -> float:
    '''
    Calculates the information gain of the given feature. 
    
    Parameters
    ----------
    col_values: ArrayLike
        Array with values of a certain feature.
    Y: ArrayLike
        Array with target values.
        
    Returns 
    -------
    information_gain: float
        Information gain value of the given feature. 
    '''
    pre_entropy = entropy(Y)
    unique_values = np.unique(col_values)
    post_entropy = 0
    for value in unique_values:
        Y_filtered = [y for i, y in enumerate(Y) if col_values[i] == value]
        post_entropy += (len(Y_filtered) / len(Y)) * entropy(Y_filtered)
    return pre_entropy - post_entropy
    
def max_information_gain(X: MatrixLike, Y: ArrayLike) -> tuple[int, float]:
    max = 0, 0.0
    
    for i in range(len(X[0])):
        ig = information_gain(X[:, i], Y)
        if ig > max[1]:
            max = i, ig
    
    return max
    
    
def c4_5(X: MatrixLike, Y: ArrayLike):
    raise NotImplementedError