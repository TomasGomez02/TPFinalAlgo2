import numpy as np
from treeModels._typing import *
from enum import Enum
from treeModels.base_tree import DecisionTree, CategoricDecision, NumericDecision
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
    maxim = 0, 0.0
    for i in range(len(X[0])):
        ig = information_gain(X[:, i], Y)       #type: ignore
        if ig > maxim[1]:
            maxim = i, ig
    
    return maxim

def id3(current_node: DecisionTree, params: dict, labels: ArrayLike, current_height: int = 1):
    """
    Recursively builds a decision tree using the ID3 algorithm.

    Parameters
    ----------
    current_node : DecisionTree
        The current node of the decision tree being constructed.
    params : dict
        Parameters for controlling the tree growth, including:
            - 'max_depth': int, the maximum depth of the tree.
            - 'min_samples_split': int, the minimum number of samples required to split an internal node.
            - 'min_samples_leaf': int, the minimum number of samples required to be at a leaf node.
            - 'min_impurity_decrease': float, the minimum information gain required to make a split.
    labels : ArrayLike
        An array-like structure containing the names or labels of the features.
    current_height : int, optional
        The current height of the tree during the construction (default is 1).

    Returns
    -------
    None
        This function modifies the tree in place and does not return a value.
    """
    max_ig_idx, info_gain = max_information_gain(current_node.samples, current_node.target)
    least_common_amount = Counter(current_node.samples[:, max_ig_idx]).most_common()[-1][1]
    if params['max_depth'] <= current_height or params['min_samples_split'] > len(current_node.samples) or params['min_samples_leaf'] > least_common_amount or params['min_impurity_decrease'] > info_gain or info_gain == 0.0:
        return
    
    current_node.decision = CategoricDecision(max_ig_idx, labels[max_ig_idx])
    for col_value in np.unique(current_node.samples[:, max_ig_idx]):
        filter_values = current_node.samples[:, max_ig_idx] == col_value
        filtered_samples = current_node.samples[filter_values]
        filtered_target = current_node.target[filter_values]
        new_tree = DecisionTree(filtered_samples, filtered_target, current_node.classes)
        current_node.insert_tree(col_value, new_tree)
        id3(new_tree, params, labels, current_height + 1)
            
def split_info(col_values: ArrayLike, Y: ArrayLike) -> float:
    """
    Calculates the split information (or intrinsic value) for a given feature column.

    Parameters
    ----------
    col_values : ArrayLike
        The values of the feature column being evaluated.
    Y : ArrayLike
        The target values corresponding to the feature column.

    Returns
    -------
    split_info : float
        The calculated split information value.
    """
    unique_values = np.unique(col_values)
    _split_info = 0
    for value in unique_values:
        Y_filtered = [y for i, y in enumerate(Y) if col_values[i] == value]
        value_ratio = len(Y_filtered) / len(col_values)
        _split_info += value_ratio * np.log2(value_ratio)
    return -_split_info
            
def gain_ratio(col_values: ArrayLike, Y: ArrayLike) -> float:
    """
    Calculates the gain ratio for a given feature column.

    Parameters
    ----------
    col_values : ArrayLike
        The values of the feature column being evaluated.
    Y : ArrayLike
        The target values corresponding to the feature column.

    Returns
    -------
    gain_ratio : float
        The calculated gain ratio value.
    """
    si = split_info(col_values, Y)
    return float(information_gain(col_values, Y)) / float(si if si != 0 else 1)

def get_umbral_candidates(X: ArrayLike, Y: ArrayLike) -> list[float]:
    """
    Generates threshold candidates for a given feature column based on the target values.

    Parameters
    ----------
    X : ArrayLike
        The feature column values for which threshold candidates are to be generated.
    Y : ArrayLike
        The target values corresponding to the feature column.

    Returns
    -------
    candidates : list of float
        A list of threshold candidates.
    """
    candidates = []
    for i in range(len(X) - 1):
        y = i + 1
        if Y[i] != Y[y] and X[i] != X[y]:
            candidates.append((X[i] + X[y]) / 2)
    return candidates

def get_umbral(X: ArrayLike, Y: ArrayLike) -> float:
    """
    Finds the best threshold (umbral) for a given feature column that maximizes information gain.

    Parameters
    ----------
    X : ArrayLike
        The feature column values for which the best threshold is to be found.
    Y : ArrayLike
        The target values corresponding to the feature column.

    Returns
    -------
    best_umbral : float
        The best threshold value that maximizes information gain.
    """
    ordered_indices = np.argsort(X)
    X, Y = X[ordered_indices], Y[ordered_indices]
    
    max_ig, best_umbral = 0.0, 0
    candidates = get_umbral_candidates(X, Y)
    for candidate in candidates:
        ig = information_gain([0 if x <= candidate else 1 for x in X] , Y)                  #type: ignore
        if ig > max_ig:
            max_ig, best_umbral = ig, candidate
    return best_umbral
    
def max_gain_ratio(X: MatrixLike, Y: ArrayLike) -> tuple[int, float]:
    """
    Finds the feature column with the maximum gain ratio.

    Parameters
    ----------
    X : MatrixLike
        Matrix-like array with feature values, where each column represents a feature.
    Y : ArrayLike
        Array with target values.

    Returns
    -------
    maxim : tuple[int, float]
        A tuple containing the index of the feature with the highest gain ratio and the gain ratio value.
    """
    maxim = 0, 0.0
    for i in range(len(X[0])):
        gr = gain_ratio(X[:, i], Y)             #type: ignore
        if gr > maxim[1]:
            maxim = i, gr
            
    return maxim

class ColumnTypes(Enum):
    """
    Enum representing the types of columns in a dataset.

    Attributes
    ----------
    NUMERIC : str
        Represents numeric columns.
    CATEGORIC : str
        Represents categorical columns.
    """
    NUMERIC = 'Numeric'
    CATEGORIC = 'Categoric'
    
def get_col_types(X: MatrixLike) -> list[ColumnTypes]:
    """
    Determines the types of columns in a dataset.

    Parameters
    ----------
    X : MatrixLike
        The dataset represented as a matrix-like array, where each column is a feature.

    Returns
    -------
    types : list of ColumnTypes
        A list containing the type of each column in the dataset.
    """
    types = []
    for col_i in range(X.shape[1]):
        e = X[0, col_i]
        if isinstance(e, int) or isinstance(e, float):
            types.append(ColumnTypes.NUMERIC)
        else:
            types.append(ColumnTypes.CATEGORIC)
    return types

def create_categoric_matrix(X: MatrixLike, Y: ArrayLike, types: list[ColumnTypes]) -> tuple[MatrixLike, dict[int, float]]:
    """
    Transforms a dataset into a categoric matrix based on column types and corresponding thresholds.

    Parameters
    ----------
    X : MatrixLike
        The dataset represented as a matrix-like array, where each column is a feature.
    Y : ArrayLike
        The target values corresponding to the feature columns.
    types : list[ColumnTypes]
        A list of column types indicating whether each column is numeric or categoric.

    Returns
    -------
    categoric_matrix : MatrixLike
        The transformed categoric matrix where numeric columns are binarized based on thresholds.
    umbrals : dict[int, float]
        A dictionary mapping column indices to their respective thresholds for numeric columns.
    """
    categoric_matrix = []
    umbrals = {}
    for col_i in range(X.shape[1]):
        col = X[:, col_i]
        if types[col_i] == ColumnTypes.NUMERIC:
            umbral = get_umbral(col, Y)
            umbrals[col_i] = umbral
            categoric_matrix.append([0 if x <= umbral else 1 for x in col])     #type: ignore
        else:
            categoric_matrix.append(col)
    return np.array(categoric_matrix).T, umbrals

def c45(current_node: DecisionTree, params: dict, labels: ArrayLike, current_height: int = 1):
    """
    Constructs a decision tree using the C4.5 algorithm.

    Parameters
    ----------
    current_node : DecisionTree
        The current node in the decision tree where the decision and splits will be made.
    params : dict
        A dictionary of parameters for the tree construction, including:
            - 'max_depth': int
                The maximum depth the tree is allowed to grow.
            - 'min_samples_split': int
                The minimum number of samples required to split an internal node.
            - 'min_samples_leaf': int
                The minimum number of samples that a leaf node must have.
            - 'min_impurity_decrease': float
                The minimum impurity decrease required to perform a split.
    labels : ArrayLike
        The labels or names of the features used in the dataset.
    current_height : int, optional
        The current depth or height of the node in the tree, by default 1.

    Returns
    -------
    None
        This function modifies the tree in place by adding nodes and decisions.

    Notes
    -----
    The C4.5 algorithm is an extension of the ID3 algorithm that handles both categorical and continuous data.
    It creates a decision tree based on the highest gain ratio calculated from the features.

    """
    types = get_col_types(current_node.samples)
    categoric_matrix, umbrals = create_categoric_matrix(current_node.samples, current_node.target, types)
    max_rg_idx, _gain_ratio = max_gain_ratio(categoric_matrix, current_node.target)
    least_common_amount = Counter(categoric_matrix[:, max_rg_idx]).most_common()[-1][1]
    if params['max_depth'] <= current_height or params['min_samples_split'] > len(current_node.samples) or params['min_samples_leaf'] > least_common_amount or params['min_impurity_decrease'] > _gain_ratio or _gain_ratio == 0.0:
        return
    
    if types[max_rg_idx] == ColumnTypes.NUMERIC:
        umbral = umbrals[max_rg_idx]
        current_node.decision = NumericDecision(max_rg_idx, umbral, labels[max_rg_idx])
        
        filter_index = current_node.samples[:, max_rg_idx] <= umbral
        filtered_samples = current_node.samples[filter_index]
        filtered_target = current_node.target[filter_index]
        smaller_tree = DecisionTree(filtered_samples, filtered_target, current_node.classes)
        current_node.insert_tree(current_node.decision.values[0], smaller_tree)
        c45(smaller_tree, params, labels, current_height + 1)
        
        filter_index = current_node.samples[:, max_rg_idx] > umbral
        filtered_samples = current_node.samples[filter_index]
        filtered_target = current_node.target[filter_index]
        larger_tree = DecisionTree(filtered_samples, filtered_target, current_node.classes)
        current_node.insert_tree(current_node.decision.values[1], larger_tree)
        c45(larger_tree, params, labels, current_height + 1)
    else:
        current_node.decision = CategoricDecision(max_rg_idx, labels[max_rg_idx])
        for col_value in np.unique(current_node.samples[:, max_rg_idx]):
            filter_values = current_node.samples[:, max_rg_idx] == col_value
            filtered_samples = current_node.samples[filter_values]
            filtered_target = current_node.target[filter_values]
            new_tree = DecisionTree(filtered_samples, filtered_target, current_node.classes)
            current_node.insert_tree(col_value, new_tree)
            c45(new_tree, params, labels,  current_height + 1)
      
class DecisionAlgorithm(Enum):
    """
    An enumeration for decision tree algorithms, providing partial functions
    for ID3 and C4.5 algorithms.

    Attributes
    ----------
    ID3 : partial
        A partial function that applies the ID3 decision tree algorithm.
    C45 : partial
        A partial function that applies the C4.5 decision tree algorithm.
    """
    ID3 = partial(id3)
    C45 = partial(c45)
    
    def __call__(self, *args):
        self.value(*args)