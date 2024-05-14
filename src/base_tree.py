from _typing import *
from abc import ABC, abstractmethod

class BaseDecision(ABC):
    def __init__(self, atr_indx: int):
        self.atr_indx = atr_indx
    
    @abstractmethod
    def make_choice(self, X: ArrayLike) -> str:
        raise NotImplementedError()
    
class NumericDecision(BaseDecision):
    def __init__(self, atr_indx, value: float | int):
        super().__init__(atr_indx)
        self.value = value
        self.values = [f'<={value}', f'>{value}']
    
    def make_choice(self, X: ArrayLike) -> str:
        if X[self.atr_indx] <= self.value:
            return self.values[0]
        return self.values[1]
        
class CategoricDecision(BaseDecision):
    def make_choice(self, X: ArrayLike) -> str:
        return X[self.atr_indx]

class BaseTree:
    def __init__(self, value, data):
        self.value: BaseDecision | str = value
        self.data = data
        self.forest: dict[str, BaseTree] = dict()
        
    def is_leaf(self):
        return self.forest == {}
    
    def height(self):
        if self.is_leaf():
            return 1
        else:
            return 1 + max([self.forest[key].height() for key in self.forest.keys()])
        
    def insert_tree(self, value: str, tree: "BaseTree"):
        self.forest[value] = tree
        
    def walkthrough(self, X: ArrayLike):
        if self.is_leaf():
            return self.value
        return self.forest[self.value.make_choice(X)].walkthrough(X)