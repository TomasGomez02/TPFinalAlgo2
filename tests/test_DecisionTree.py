from treeModels import DecisionTreeClassifier as TreeClassifierTM
from sklearn.tree import DecisionTreeClassifier as TreeClassifierSKL
from sklearn.preprocessing import LabelEncoder
import pytest
import pandas as pd
from collections import Counter


def test_overfitted(X, Y):
    model = TreeClassifierTM()
    model.fit(X.iloc[2:, :], Y[2:])
    assert model.get_depth() == 5
    assert Counter(model.predict(X.loc[0:2,:])) == {"No":2, "Yes":1}
    assert model.predict_proba(X).shape == (19, 2)
    assert round(model.predict_proba(X.iloc[2:,:]).max(), 2) == 1
    assert TreeClassifierTM().fit(X.iloc[0:1,:], Y.iloc[0:1]).score(X.iloc[0:1,:], Y.iloc[0:1]) == 1
    with pytest.raises(NotImplementedError):
        model.decision_path(X)

def test_params(X, Y):
    assert TreeClassifierTM(max_depth=3).fit(X, Y).get_depth() == 3
    assert len(TreeClassifierTM(min_samples_split=5).fit(X, Y).tree.samples) >= 5
    params = {"max_depth":200, "min_samples_split":9, "min_samples_leaf":10,
    "min_impurity_decrease": 1, "algorithm":"C4.5"}
    assert TreeClassifierTM().set_params(**params).get_params() == params

@pytest.mark.xfail
def test_C4_5():
    df = pd.read_csv("CarEval.csv")
    X = df.drop("class values", axis=1)
    Y = df["class values"]
    model = TreeClassifierTM(algorithm="C4.5").fit(X.iloc[5:,:], Y.iloc[5:])
    assert Counter(model.predict(X.iloc[:5,:])) == {"unacc":5}
    assert round(model.predict_proba(X.iloc[:5,:]).max(), 2) == 1
    

def test_Exceptions_Errors(X, Y):
    overFittedTree = TreeClassifierTM()
    with pytest.raises(ValueError):
        overFittedTree.predict(X)
    overFittedTree.set_params(nombre="gonza")
    with pytest.raises(AttributeError):
        overFittedTree.nombre