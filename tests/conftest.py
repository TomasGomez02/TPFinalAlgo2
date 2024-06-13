import pytest
from treeModels import DecisionTreeClassifier
import pandas
from sklearn import tree
from sklearn.preprocessing import LabelEncoder

@pytest.fixture(scope="session")
def overFittedTreeTM() -> DecisionTreeClassifier:
    df = pandas.read_csv("CarEval.csv")
    X = df.drop("play", axis=1)
    Y = df["play"]
    tree = DecisionTreeClassifier()
    return tree.fit(X, Y)

@pytest.fixture(scope="session")
def overFittedTreeSK() -> tree.DecisionTreeClassifier:
    df = pandas.read_csv("CarEval.csv")
    df["doors"] = str(df["doors"])
    df["persons"] = str(df["persons"])
    
    for column in df:
        if df[column].dtype == object:
            df[column] = LabelEncoder().fit_transform(df[column])
    overFitted = tree.DecisionTreeClassifier().fit(df.drop("class values", axis=1), df["class values"])
    return overFitted

@pytest.fixture(scope="session")
def DF() -> pandas.DataFrame:
    df = pandas.read_csv("play_tennis.csv")
    return df

@pytest.fixture(scope="session")
def X(DF) -> pandas.DataFrame:
    return DF.drop("play", axis=1)

@pytest.fixture(scope="session")
def Y(DF) -> pandas.DataFrame:
    return DF["play"]