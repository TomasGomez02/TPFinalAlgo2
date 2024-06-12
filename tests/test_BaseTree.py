import pytest
from treeModels.base_tree import NumericDecision, BaseDecision
from treeModels import DecisionTreeClassifier

def test_tree(X, Y):
    overFitted = DecisionTreeClassifier().fit(X, Y)
    assert overFitted.tree.copy().get_n_leaves() == overFitted.get_n_leaves()
    assert overFitted.tree.get_class() == "Yes"
    assert overFitted.tree.get_n_leaves() == 11
    assert overFitted.tree.to_leaf().height() == 1

