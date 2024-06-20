from treeModels._typing import *
from abc import ABC, abstractmethod
from typing import Optional, Any
from collections import Counter


class BaseDecision(ABC):
    """
    Abstract base class for decision-making based on a specific attribute index.

    Parameters
    ----------
    atr_indx : int
        Index of the attribute based on which the decision is made.
    """
    def __init__(self, atr_indx: int, atr_label: str):
        self.atr_indx = atr_indx
        if not isinstance(atr_label, str):
            atr_label = str(atr_label)
        self.atr_label = atr_label
    
    @abstractmethod
    def make_choice(self, X: ArrayLike) -> str:
        """
        Makes a decision based on the attribute at the given index.

        Parameters
        ----------
        X : ArrayLike
            Array-like structure containing the data for making the decision.

        Returns
        -------
        choice : str
            The decision made based on the attribute.
        """
        raise NotImplementedError()
    
    @abstractmethod
    def copy(self) -> "BaseDecision":
        """
        Creates a copy of the decision-making instance.

        Returns
        -------
        decision_copy : BaseDecision
            A copy of the current decision-making instance.
        """
        raise NotImplementedError()
    
class NumericDecision(BaseDecision):
    """
    Decision-making class for numerical attributes based on a threshold value.

    Parameters
    ----------
    atr_indx : int
        Index of the attribute based on which the decision is made.
    value : float or int
        Threshold value for making the decision.
    """
    def __init__(self, atr_indx, value: float | int, atr_label: str):
        super().__init__(atr_indx, atr_label)
        self.value = value
        self.values = [f'<={value}', f'>{value}']
    
    def make_choice(self, X: ArrayLike) -> str:
        """
        Makes a decision based on whether the attribute value is less than or equal to the threshold.

        Parameters
        ----------
        X : ArrayLike
            Array-like structure containing the data for making the decision.

        Returns
        -------
        choice : str
            The decision made based on whether the attribute value is less than or equal to the threshold.
        """
        if X[self.atr_indx] <= self.value:
            return self.values[0]
        return self.values[1]
    
    def copy(self) -> "NumericDecision":
        """
        Creates a copy of the NumericDecision instance.

        Returns
        -------
        decision_copy : NumericDecision
            A copy of the current NumericDecision instance.
        """
        new = NumericDecision(self.atr_indx, self.value, self.atr_label)
        return new
        
class CategoricDecision(BaseDecision):
    """
    Decision-making class for categorical attributes.

    Parameters
    ----------
    atr_indx : int
        Index of the attribute based on which the decision is made.
    """
    def make_choice(self, X: ArrayLike) -> str:
        """
        Makes a decision based on the categorical value of the attribute at the given index.

        Parameters
        ----------
        X : ArrayLike
            Array-like structure containing the data for making the decision.

        Returns
        -------
        choice : str
            The categorical value of the attribute at the specified index.
        """
        return X[self.atr_indx]
    
    def copy(self) -> "CategoricDecision":
        """
        Creates a copy of the CategoricDecision instance.

        Returns
        -------
        decision_copy : CategoricDecision
            A copy of the current CategoricDecision instance.
        """
        new = CategoricDecision(self.atr_indx, self.atr_label)
        return new

class DecisionTree:
    def __init__(self, samples: MatrixLike, target: ArrayLike, classes: ArrayLike):
        """
        A tree structure for making decisions based on the given samples and target values.

    Attributes
    ----------
    samples : np.ndarray
        The samples at the current node.
    target : np.ndarray
        The target values associated with the samples.
    classes : np.ndarray
        The unique classes in the target values.
    decision : Optional[BaseDecision]
        The decision at the current node.
    forest : Dict[str, BaseTree]
        The subtrees of the current node.
    """
    
    def __init__(self, samples: MatrixLike, target: ArrayLike, classes: ArrayLike):
        self.decision: Optional[BaseDecision] = None
        self.samples = samples
        self.target = target
        self.classes = classes
        self.forest: dict[str, DecisionTree] = dict()
        
    def is_leaf(self):
        """
        Checks if the current node is a leaf node.

        Returns
        -------
        is_leaf : bool
            True if the current node is a leaf, False otherwise.
        """
        return self.forest == {}
    
    def height(self):
        """
        Computes the height of the tree.

        Returns
        -------
        height : int
            The height of the tree.
        """
        if self.is_leaf():
            return 1
        else:
            return 1 + max([self.forest[key].height() for key in self.forest.keys()])
        
    def insert_tree(self, value: str, tree: "DecisionTree"):
        """
        Inserts a subtree into the current tree.

        Parameters
        ----------
        value : str
            The value associated with the subtree.
        tree : BaseTree
            The subtree to be inserted.
        """
        self.forest[value] = tree
    
    def get_class(self) -> Any:
        """
        Gets the most common class in the target values.

        Returns
        -------
        most_common_class : Any
            The most common class in the target values.
        """
        count = Counter(self.target)
        return count.most_common(1)[0][0]
        
    def walkthrough(self, X: ArrayLike):
        """
        Walks through the tree to make a prediction based on the given input.

        Parameters
        ----------
        X : ArrayLike
            An array-like structure containing the input data.

        Returns
        -------
        prediction : Any
            The predicted class for the given input.
        """
        if self.is_leaf() or (not X[self.decision.atr_indx] in self.forest.keys() and not isinstance(self.decision, NumericDecision)):
            return self.get_class()
        return self.forest[self.decision.make_choice(X)].walkthrough(X)
    
    def walkthrough_proba(self, X: ArrayLike):
        """
        Walks through the tree to get the class proportions based on the given input.

        Parameters
        ----------
        X : ArrayLike
            An array-like structure containing the input data.

        Returns
        -------
        class_proportions : np.ndarray
            The class proportions for the given input.
        """
        if self.is_leaf() or not X[self.decision.atr_indx] in self.forest.keys():
            return self.get_class_proportion()
        return self.forest[self.decision.make_choice(X)].walkthrough_proba(X)
    
    def get_class_proportion(self):
        """
        Gets the proportions of each class in the target values.

        Returns
        -------
        class_proportions : np.ndarray
            An array of class proportions.
        """
        count = {cls: 0 for cls in self.classes}
        for y in self.target:
            count[y] += 1
        return np.array(list(count.values())) / len(self.target)
    
    def copy(self) -> "DecisionTree":
        """
        Creates a copy of the current tree.

        Returns
        -------
        tree_copy : BaseTree
            A copy of the current tree.
        """
        new = DecisionTree(self.samples, self.target, self.classes)
        if not self.is_leaf():
            new.forest = self.forest.copy()
            new.decision = self.decision.copy()
        return new
    
    def to_leaf(self) -> "DecisionTree":
        """
        Converts the current tree to a leaf node.

        Returns
        -------
        leaf : BaseTree
            A copy of the current tree as a leaf node.
        """
        leaf = self.copy()
        if not self.is_leaf():
            leaf.forest = dict()
            leaf.decision = None
        return leaf
    
    def get_n_leaves(self) -> int:
        """
        Computes the number of leaf nodes in the tree.

        Returns
        -------
        n_leaves : int
            The number of leaf nodes in the tree.
        """
        if self.is_leaf():
            return 1
        leaves = 0
        for key in self.forest.keys():
            leaves += self.forest[key].get_n_leaves()
        return leaves
    
    def n_samples(self) -> int:
        """
        Gets the number of samples in the current node.

        Returns
        -------
        n_samples : int
            The number of samples in the current node.
        """
        return self.samples.shape[0]
    
    def __str__(self):
        """
        Returns a string representation of the tree.

        Returns
        -------
        tree_str : str
            A string representation of the tree.
        """
        def mostrar(t: DecisionTree, nivel: int, value_name = ''):
            tab = '.' * 4
            indent = tab * nivel
            out = indent + str(value_name) + ' | '
            if t.is_leaf():
                out += str(t.get_class()) + '\n'
            else:
                # out += str(t.decision.atr_indx) + '\n'
                out += f"[{t.decision.atr_label}]" + '\n'
            for key in t.forest.keys():
                out += mostrar(t.forest[key], nivel + 1, key)
            return out
            
        return mostrar(self, 0)
    
    def __repr__(self):
        return self.__str__()

    def get_label(self) -> str:
        """
        Gets the attribute label of the decision at the current node.

        Returns
        -------
        label : str
            The attribute label of the decision.
        """
        return self.decision.atr_label
    
    def set_labels(self, labels: list):
        """
        Sets the attribute labels for the decisions in the tree.

        Parameters
        ----------
        labels : list
            A list of attribute labels corresponding to the indices of the features 
            used in the decisions.
        """
        if not self.is_leaf():
            self.decision.atr_label = labels[self.decision.atr_indx]
            for key in self.forest.keys():
                self.forest[key].set_labels(labels)
    
    def get_classes(self) -> ArrayLike:
        """
        Returns a copy of the classes array.

        Returns
        -------
        classes_copy : np.ndarray
            A copy of the array of unique classes.
        """
        return self.classes.copy()
    
    def get_impurity(self) -> float:
        """
        Computes the impurity of the target values at the current node.

        Returns
        -------
        impurity : float
            The impurity of the target values at the current node.
        """
        from treeModels.decision_algorithms import entropy
        return entropy(self.target)