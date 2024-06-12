from treeModels._typing import MatrixLike, ArrayLike
from scipy.sparse import spmatrix
from numpy import ndarray
import numpy as np
from typing import Optional, Literal, Any
from dataclasses import dataclass
from treeModels.decision_algorithms import DecisionAlgorithm
from treeModels.base_tree import BaseTree
from collections import Counter
from treeModels.models.model import Model

@dataclass(repr=False, eq=False)
class DecisionTreeClassifier(Model):
    max_depth: Optional[int | float] = np.inf
    min_samples_split: Optional[int | float] = 2
    min_samples_leaf: Optional[int | float] = 1
    min_impurity_decrease: Optional[float] = 0.0
    algorithm: DecisionAlgorithm = DecisionAlgorithm.ID3
        
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
        self.tree = BaseTree(np.array(X), np.array(Y), np.unique(Y))
        self.algorithm(self.tree, self.get_params())
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
        return self.tree.get_n_leaves()
        
    def get_params(self) -> dict:
        params = self.__dict__.copy()
        if "tree" in params.keys():
            params.pop("tree")
        return params

@dataclass(repr=False, eq=False)
class RandomForestClassifier(Model):
    n_estimators: int = 100
    max_depth: Optional[int | float] = np.inf
    min_samples_split: Optional[int | float] = 2
    min_samples_leaf: Optional[int | float] = 1
    min_impurity_decrease: Optional[float] = 0.0
    algorithm: DecisionAlgorithm = DecisionAlgorithm.ID3
    bootstrap: bool = True
    max_features: Literal['sqrt', 'log2', None] = 'sqrt'
    max_samples: int | float | None = None
    random_state: int | None = None
    
    def _plant_forest(self):
        self.forest = [DecisionTreeClassifier(self.max_depth, self.min_samples_split, self.min_samples_leaf, self.min_impurity_decrease, self.algorithm) for i in range(self.n_estimators)]
    
    def _generate_random_samples(self, X: MatrixLike, Y: ArrayLike, random_generator: Optional[np.random.RandomState] = None) -> tuple[MatrixLike, ArrayLike]:
        n_samples = X.shape[0]
        if isinstance(self.max_samples, int):
            n_samples = self.max_samples
        elif isinstance(self.max_samples, float):
            n_samples = round(n_samples * max(min(self.max_samples, 1.0), 0.0))
            
        indices = [i for i in range(len(X))]
        
        if random_generator is None:
            random_indices = np.random.choice(indices, n_samples, True)
        else:
            random_indices = random_generator.choice(indices, n_samples, True)
        
        random_X = [X[i] for i in random_indices]
        random_Y = [Y[i] for i in random_indices]
        return np.array(random_X), np.array(random_Y)
    
    def _generate_random_features(self, n_cols, random_generator: Optional[np.random.RandomState] = None) -> ndarray:
        if self.max_features == 'sqrt':
            n_features = round(np.sqrt(n_cols))
        elif self.max_features == 'log2':
            n_features = round(np.log2(n_cols))
        else:
            n_features = n_cols
            
        indices = [i for i in range(n_cols)]
        
        if n_features == n_cols:
            return np.array(indices)
        
        np.sort(np.random.choice(indices, n_features, False))
        if random_generator is None:
            random_features = np.random.choice(indices, n_features, False)
        else:
            random_features = random_generator.choice(indices, n_features, False)
        
        return np.sort(random_features)
    
    def _select_features(self, X: MatrixLike, features: ArrayLike) -> MatrixLike:
        return np.array([X[:,i] if i in features else np.array(['' for index in range(X.shape[0])]) for i in range(X.shape[1])]).T
            
    def fit(self, X: MatrixLike, Y: ArrayLike) -> "RandomForestClassifier":
        self._plant_forest()
        
        X_array = np.array(X)
        Y_array = np.array(Y)
        
        random_generator = np.random.RandomState(seed=self.random_state) if self.random_state is not None else None
        
        for i in range(self.n_estimators):
            if self.bootstrap:
                X_random_sample, Y_random_sample = self._generate_random_samples(X_array, Y_array, random_generator)
            else:
                X_random_sample, Y_random_sample = X_array, Y_array
            
            features = self._generate_random_features(X_array.shape[1], random_generator)

            X_random_sample = self._select_features(X_random_sample, features)

            self.forest[i].fit(X_random_sample, Y_random_sample)
        
        return self
    
    def predict(self, X: MatrixLike) -> ndarray:
        if not hasattr(self, "forest"):
            raise ValueError('You must call fit() method first.')
        X = np.array(X)
        if len(X.shape) == 1:
            predictions = [tree.predict(X).item() for tree in self.forest]
            counter = Counter(predictions)
            return np.array(counter.most_common(1)[0][0])
        else:
            res = []
            for row in X:
                res.append(self.predict(row))
            return np.array(res)
    
    def predict_proba(self, X: MatrixLike) -> ndarray:
        if not hasattr(self, "forest"):
            raise ValueError('You must call fit() method first.')
        X = np.array(X)
        if len(X.shape) == 1:
            predictions = [tree.predict_proba(X) for tree in self.forest]
            return np.mean(np.array(predictions), axis=0)
        else:
            res = []
            for row in X:
                res.append(self.predict(row))
            return np.array(res)
    
    def set_params(self, **params) -> "RandomForestClassifier":
        for key in params.keys():
            if hasattr(self, key):
                self.__setattr__(key, params[key])
        return self
    
    def score(self, X: MatrixLike, Y: ArrayLike) -> float:
        Y_predict = self.predict(X)
        return np.sum(Y_predict == Y) / len(Y)
    
    def decision_path(self, X: MatrixLike) -> spmatrix:
        raise NotImplementedError
    
    #TODO: para gonza
    def get_params(self) -> dict: 
        raise NotImplementedError

def main():
    pass

if __name__ == '__main__':
    main()