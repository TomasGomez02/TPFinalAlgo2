import numpy as np
from treeModels._typing import *
from enum import Enum
from treeModels.base_tree import BaseTree, CategoricDecision, NumericDecision
from collections import Counter
from functools import partial

def entropy(Y: ArrayLike) -> float:
    '''
    Calculates the entropy of a given target array.
    
    Parameters
    ----------
    Y: ArrayLike 
        Target column to calculate the entropy of.
        
    Returns
    -------
    entropy: 
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
    '''
    Calculates the information gain for each feature in the dataset and returns the feature with the maximum information gain.
    
    Parameters
    ----------
    X: MatrixLike
        Matrix-like array with feature values, where each column represents a feature.
    Y: ArrayLike
        Array with target values.
        
    Returns 
    -------
    max_information_gain: tuple[int, float]
        A tuple containing the index of the feature with the highest information gain and the information gain value. 
    '''
    max = 0, 0.0
    for i in range(len(X[0])):
        ig = information_gain(X[:, i], Y)
        if ig > max[1]:
            max = i, ig
    
    return max

def id3(current_node: BaseTree, params: dict, labels: ArrayLike, current_height: int = 1):
        max_ig_idx, info_gain = max_information_gain(current_node.samples, current_node.target)
        least_common_amount = Counter(current_node.samples[:, max_ig_idx]).most_common()[-1][1]
        if params['max_depth'] <= current_height or params['min_samples_split'] > len(current_node.samples) or params['min_samples_leaf'] > least_common_amount or params['min_impurity_decrease'] > info_gain or info_gain == 0.0:
            return
        
        current_node.decision = CategoricDecision(max_ig_idx, labels[max_ig_idx])
        for col_value in np.unique(current_node.samples[:, max_ig_idx]):
            filter_values = current_node.samples[:, max_ig_idx] == col_value
            filtered_samples = current_node.samples[filter_values]
            filtered_target = current_node.target[filter_values]
            new_tree = BaseTree(filtered_samples, filtered_target, current_node.classes)
            current_node.insert_tree(col_value, new_tree)
            id3(new_tree, params, labels, current_height + 1)
            
def c45(current_node: BaseTree, params: dict, labels: ArrayLike, current_height: int = 1):
    raise NotImplementedError
            
class DecisionAlgorithm(Enum):
    ID3 = partial(id3)
    C45 = partial(c45)
    
    def __call__(self, *args):
        self.value(*args)