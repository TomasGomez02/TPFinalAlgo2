from treeModels import DecisionTreeClassifier
import pytest

def test_imports():
    from treeModels import BaseTree
    from treeModels import CategoricDecision
    from treeModels import DecisionTreeClassifier

def test_TreeClassifier(overFittedTree):
    assert overFittedTree.get_depth() == 5