from _typing import MatrixLike, ArrayLike
from scipy.sparse import spmatrix
from numpy import ndarray
from typing import Optional, Literal

class DecisionTreeClassifier:
    def __init__(self, max_depth: Optional[int] = None, 
                 min_samples_split: Optional[int | float] = 2,
                 min_samples_leaf: Optional[int | float] = 1,
                 min_impurity_decrease: Optional[float] = 0.0,
                 algorithm: Literal['ID3', 'C4.5'] = 'ID3'):
        raise NotImplementedError
    
    def fit(self, X: MatrixLike, Y: MatrixLike) -> "DecisionTreeClassifier":
        raise NotImplementedError
    
    def predict(self, X: MatrixLike) -> ndarray:
        raise NotImplementedError
    
    def score(self, X: MatrixLike, Y: MatrixLike) -> float:
        raise NotImplementedError

class RandomForestClassifier:
    def __init__(self):
        raise NotImplementedError
    
    def fit(self, X: MatrixLike, Y: MatrixLike) -> "RandomForestClassifier":
        raise NotImplementedError
    
    def predict(self, X: MatrixLike) -> ndarray:
        raise NotImplementedError
    
    def predict_proba(self, X: MatrixLike) -> ndarray:
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