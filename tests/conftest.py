import pytest
from treeModels import DecisionTreeClassifier
import pandas

@pytest.fixture(scope="session")
def overFittedTree():
    df = pandas.read_csv("play_tennis.csv")
    X = df.drop("play", axis=1)
    Y = df["play"]
    tree = DecisionTreeClassifier()
    return tree.fit(X, Y)

@pytest.fixture(scope="session")
def DF():
    df = pandas.read_csv("play_tennis.csv")
    return df

@pytest.fixture(scope="session")
def X(DF):
    return DF.drop("play", axis=1)

@pytest.fixture(scope="session")
def Y(DF):
    return DF["play"]