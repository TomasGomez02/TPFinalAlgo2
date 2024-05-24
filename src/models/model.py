from abc import ABC, abstractmethod
from _typing import *

class Model(ABC):
    @abstractmethod
    def fit(self, X, Y) -> "Model":
        raise NotImplementedError()
    
    @abstractmethod
    def predict(self, X) -> ArrayLike:
        raise NotImplementedError()
    
    @abstractmethod
    def predict_proba(self, X) -> ArrayLike:
        raise NotImplementedError()
    
    @abstractmethod
    def score(self, X, Y) -> float:
        raise NotImplementedError()