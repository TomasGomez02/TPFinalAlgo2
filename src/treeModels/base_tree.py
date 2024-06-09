from treeModels._typing import *
from abc import ABC, abstractmethod
from typing import Optional, Any
from collections import Counter

class BaseDecision(ABC):
    def __init__(self, atr_indx: int):
        self.atr_indx = atr_indx
    
    @abstractmethod
    def make_choice(self, X: ArrayLike) -> str:
        raise NotImplementedError()
    
    @abstractmethod
    def copy(self) -> "BaseDecision":
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
    
    def copy(self) -> "NumericDecision":
        new = NumericDecision(self.atr_indx, self.value)
        return new
        
class CategoricDecision(BaseDecision):
    def make_choice(self, X: ArrayLike) -> str:
        return X[self.atr_indx]
    
    def copy(self) -> "CategoricDecision":
        new = CategoricDecision(self.atr_indx)
        return new
        

class BaseTree:
    def __init__(self, samples: MatrixLike, target: ArrayLike):
        self.decision: Optional[BaseDecision] = None
        self.samples = samples
        self.target = target
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
    
    def get_class(self) -> Any:
        count = Counter(self.target)
        return count.most_common(1)[0][0]
        
    def walkthrough(self, X: ArrayLike):
        if self.is_leaf():
            return self.get_class()
        return self.forest[self.decision.make_choice(X)].walkthrough(X)
    
    def walkthrough_proba(self, X: ArrayLike):
        if self.is_leaf():
            return self.get_class_proportion()
        return self.forest[self.decision.make_choice(X)].walkthrough_proba(X)
    
    def get_class_proportion(self):
        count =  Counter(self.target)
        return np.array(list(count.values())) / len(self.target)
    
    def copy(self) -> "BaseTree":
        new = BaseTree(self.samples, self.target)
        if not self.is_leaf():
            new.forest = self.forest.copy()
            new.decision = self.decision.copy()
        return new
    
    def to_leaf(self) -> "BaseTree":
        leaf = self.copy()
        if not self.is_leaf():
            leaf.forest = dict()
            leaf.decision = None
        return leaf
    
    def __str__(self):
        def mostrar(t: BaseTree, nivel: int, value_name = ''):
            tab = '.' * 4
            indent = tab * nivel
            out = indent + value_name + ' | '
            if t.is_leaf():
                out += str(t.get_class()) + '\n'
            else:
                out += str(t.decision.atr_indx) + '\n'
            for key in t.forest.keys():
                out += mostrar(t.forest[key], nivel + 1, key)
            return out
            
        return mostrar(self, 0)