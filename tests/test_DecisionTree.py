from treeModels import DecisionTreeClassifier
import pytest
import pandas as pd
from collections import Counter

def test_overfitted(X, Y):
    model = DecisionTreeClassifier()
    model.fit(X.iloc[2:, :], Y[2:])
    assert model.get_depth() == 5
    assert Counter(model.predict(X.loc[0:2,:])) == {"No":2, "Yes":1}
    assert model.get_n_leaves() == 11
    assert model.predict_proba(X).shape == (19, 2)

def test_params(X, Y):
    assert DecisionTreeClassifier(max_depth=3).fit(X, Y).get_depth() == 3
    assert len(DecisionTreeClassifier(min_samples_split=3)
               .fit(X, Y).tree.forest["Overcast"].samples) >= 3
    assert len(DecisionTreeClassifier(min_samples_split=5).fit(X, Y).tree.samples) >= 5
    # falta algo que vea min_samples_leaf mejor

@pytest.mark.xfail
def test_C4_5():
    df = pd.read_csv("CarEval.csv")
    X = df.drop("class values", axis=1)
    Y = df["class values"]
    model = DecisionTreeClassifier(algorithm="C4.5").fit(X.iloc[5:,:], Y.iloc[5:])
    model.predict(X.iloc[0:5,:])
    

def test_Exceptions_Errors(X, Y):
    overFittedTree = DecisionTreeClassifier()
    with pytest.raises(ValueError):
        overFittedTree.predict(X)
    overFittedTree.set_params(nombre="gonza")
    with pytest.raises(AttributeError):
        overFittedTree.nombre