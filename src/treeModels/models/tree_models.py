from treeModels._typing import MatrixLike, ArrayLike
from scipy.sparse import spmatrix
from numpy import ndarray
import numpy as np
from typing import Optional, Literal
from dataclasses import dataclass
import treeModels.decision_algorithms as da
from treeModels.base_tree import BaseTree, CategoricDecision, NumericDecision
from collections import Counter
from treeModels.models.model import Model

@dataclass
class DecisionTreeClassifier(Model):
    max_depth: Optional[int | float] = np.inf
    min_samples_split: Optional[int | float] = 2
    min_samples_leaf: Optional[int | float] = 1
    min_impurity_decrease: Optional[float] = 0.0
    algorithm: Literal['ID3', 'C4.5'] = 'ID3'
    
    def _id3(self, current_node: BaseTree, current_height: int = 1):
        '''
        Recursively builds the decision tree using the ID3 algorithm.

        Parameters
        ----------
        current_node: BaseTree
            The current node of the tree being processed.
        current_height: int, optional
            The current height (or depth) of the tree, used to control the depth of recursion.
            Defaults to 1.

        Returns
        -------
        None
            The function modifies the tree in place and does not return anything.
        '''
        max_ig_idx, info_gain = da.max_information_gain(current_node.samples, current_node.target)
        least_common_amount = Counter(current_node.samples[:, max_ig_idx]).most_common()[-1][1]
        if self.max_depth <= current_height or self.min_samples_split > len(current_node.samples) or self.min_samples_leaf > least_common_amount or self.min_impurity_decrease > info_gain or info_gain == 0.0:
            return
        
        current_node.decision = CategoricDecision(max_ig_idx)
        for col_value in np.unique(current_node.samples[:, max_ig_idx]):
            filter_values = current_node.samples[:, max_ig_idx] == col_value
            filtered_samples = current_node.samples[filter_values]
            filtered_target = current_node.target[filter_values]
            new_tree = BaseTree(filtered_samples, filtered_target)
            current_node.insert_tree(col_value, new_tree)
            self._id3(new_tree, current_height + 1)
        
    
    def fit(self, X: MatrixLike, Y: ArrayLike) -> "DecisionTreeClassifier":
        '''
        Fits the decision tree classifier to the provided data.

        Parameters
        ----------
        X: MatrixLike
            The input data matrix where each row represents a sample and each column represents a feature.
        Y: ArrayLike
            The target values corresponding to the input data.

        Returns
        -------
        self: DecisionTreeClassifier
            The fitted decision tree classifier instance.
        '''    
        self.tree = BaseTree(np.array(X), np.array(Y))
        if self.algorithm == "ID3":
            self._id3(self.tree)
        else:
            raise NotImplementedError
        return self
    
    def predict(self, X: MatrixLike) -> ndarray:
        '''
        Predicts the target values for the given input data using the fitted decision tree.

        Parameters
        ----------
        X: MatrixLike
            The input data matrix where each row represents a sample and each column represents a feature.
            If a single sample is provided, it should be a 1D array.

        Returns
        -------
        predictions: ndarray
            An array of predicted target values corresponding to the input data.

        Raises
        ------
        ValueError
            If the `fit` method has not been called before calling `predict`.
        '''    
        if not hasattr(self, "tree"):
            raise ValueError('You must call fit() method first.')
        X = np.array(X)
        if len(X.shape) == 1:
            return np.array(self.tree.walkthrough(X))
        else:
            res = []
            for row in X:
                res.append(self.predict(row))
            return np.array(res)

    def predict_proba(self, X: MatrixLike) -> ndarray:
        '''
        Predicts the probability distribution of the target values for the given input data using the fitted decision tree.

        Parameters
        ----------
        X: MatrixLike
            The input data matrix where each row represents a sample and each column represents a feature.
            If a single sample is provided, it should be a 1D array.

        Returns
        -------
        probabilities: ndarray
            An array of probability distributions corresponding to the input data. Each element in the array 
            is a list of probabilities for each class.

        Raises
        ------
        ValueError
            If the `fit` method has not been called before calling `predict_proba`.
        '''    
        if not hasattr(self, "tree"):
            raise ValueError('You must call fit() method first.')
        X = np.array(X)
        if len(X.shape) == 1:
            return np.array(self.tree.walkthrough_proba(X))
        else:
            res = []
            for row in X:
                res.append(self.predict_proba(row))
            return np.array(res)
    
    def set_params(self, **params) -> "DecisionTreeClassifier":
        '''
        Sets the parameters of the DecisionTreeClassifier.

        Parameters
        ----------
        **params: dict
            A dictionary of parameter names and their corresponding values to set in the classifier.
            
        Returns
        -------
        self: DecisionTreeClassifier
            The instance of the classifier with updated parameters.
        '''    
        for key in params.keys():
            if hasattr(self, key):
                self.__setattr__(key, params[key])
        return self
    
    def score(self, X: MatrixLike, Y: ArrayLike) -> float:
        '''
        Computes the accuracy of the classifier on the given test data and labels.

        Parameters
        ----------
        X: MatrixLike
            The input data matrix where each row represents a sample and each column represents a feature.
        Y: ArrayLike
            The true target values corresponding to the input data.

        Returns
        -------
        accuracy: float
            The accuracy of the classifier, defined as the ratio of correctly predicted samples 
            to the total number of samples.
        '''
        Y_predict = self.predict(X)
        return np.sum(Y_predict == Y) / len(Y)
    
    def prune(self, X: MatrixLike) -> "DecisionTreeClassifier":
        '''
        Prunes the decision tree to prevent overfitting by removing nodes that do not improve performance on the given data.

        Parameters
        ----------
        X: MatrixLike
            The input data matrix where each row represents a sample and each column represents a feature.

        Returns
        -------
        self: DecisionTreeClassifier
            The instance of the classifier with the pruned decision tree.
        '''
        def inner_prune(current: BaseTree, prev: BaseTree) -> None:
            has_leaf = False
            for k, sub in current.forest.items():
                if sub.is_leaf() and not has_leaf:
                    pass
                
        inner_prune(self.tree, self.tree)
        return self
    
    def decision_path(self, X: MatrixLike) -> spmatrix:
        raise NotImplementedError
    
    def get_depth(self) -> int:
        '''
        Returns the depth of the decision tree.

        Returns
        -------
        depth: int
            The depth of the decision tree, defined as the maximum depth from the root node to any leaf node.
        '''
        return self.tree.height()
    
    def get_n_leaves(self) -> int:
        '''
        Returns the number of leaves in the decision tree.

        Returns
        -------
        n_leaves: int
            The number of leaf nodes in the decision tree.

        Raises
        ------
        ValueError
            If the `fit` method has not been called before calling `get_n_leaves`.
        '''
        if not hasattr(self, "tree"):
            raise ValueError('You must call fit() method first.')    
    
    def get_params(self) -> dict:
        raise NotImplementedError

class RandomForestClassifier(Model):
    def __init__(self):
        raise NotImplementedError
    
    def fit(self, X: MatrixLike, Y: ArrayLike) -> "RandomForestClassifier":
        raise NotImplementedError
    
    def predict(self, X: MatrixLike) -> ndarray:
        raise NotImplementedError
    
    def predict_proba(self, X: MatrixLike) -> ndarray:
        raise NotImplementedError
    
    def set_params(self, **params) -> "RandomForestClassifier":
        raise NotImplementedError
    
    def score(self, X: MatrixLike, Y: ArrayLike) -> float:
        raise NotImplementedError
    
    def decision_path(self, X: MatrixLike) -> spmatrix:
        raise NotImplementedError
    
    def get_depth(self) -> int:
        raise NotImplementedError
    
    def get_n_leaves(self) -> int:
        raise NotImplementedError
    
    def get_params(self) -> dict:
        raise NotImplementedError

def main():
    pass

if __name__ == '__main__':
    main()