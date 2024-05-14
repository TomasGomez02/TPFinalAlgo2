from _typing import MatrixLike, ArrayLike
from scipy.sparse import spmatrix
from numpy import ndarray
from typing import Optional, Literal
from dataclasses import dataclass

@dataclass
class DecisionTreeClassifier:
    max_depth: Optional[int] = None 
    min_samples_split: Optional[int | float] = 2
    min_samples_leaf: Optional[int | float] = 1
    min_impurity_decrease: Optional[float] = 0.0
    algorithm: Literal['ID3', 'C4.5'] = 'ID3'
    
    def fit(self, X: MatrixLike, Y: ArrayLike) -> "DecisionTreeClassifier":
        
    
    def predict(self, X: MatrixLike) -> ndarray:
        raise NotImplementedError
    
    def predict_proba(self, X: MatrixLike) -> ndarray:
        raise NotImplementedError
    
    def set_params(self, **params) -> "DecisionTreeClassifier":
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
    
    def _create_tree(self, X: MatrixLike):
        

class RandomForestClassifier:
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