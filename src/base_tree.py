from _typing import *
from typing import Optional
from abc import ABC, abstractmethod

class BaseDecision(ABC):
    def __init__(self, attribute, values: ArrayLike):
        self.atribute = attribute
        self.values: ArrayLike = values
    
    @abstractmethod
    def make_choice(self, X: ArrayLike) -> str:
        raise NotImplementedError()
    
class NumericDecision(BaseDecision):
    def make_choice(self, X: ArrayLike) -> str:
        ...
        
class CategoricDecision(BaseDecision):
    def make_choice(self, X: ArrayLike) -> str:
        ...

class BaseTree:
    def __init__(self, value):
        self.value: BaseDecision | DecisionResult = value
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
    
    
        