import pytest
from treeModels import RandomForestClassifier
import pandas as pd
from collections import Counter

def test_overfitted():
    df = pd.read_csv("play_tennis.csv")
    X = df.drop("play", axis=1)
    Y = df["play"]
    model = RandomForestClassifier(random_state=8)
    model.fit(X, Y)
    assert Counter(model.predict(X.iloc[0:3])) == {"No":2, "Yes":1}
    with pytest.raises(NotImplementedError):
        assert model.decision_path(X.iloc[0])
    assert model.set_params(random_state=200).get_params()["random_state"] == 200
    assert round(model.score(X, Y), 2) == 0.74
    assert round(model.predict_proba(X.iloc[0,:]).max(), 2) == 0.72

# Agregar mas test para Random Forest con C4.5

def test_exceptions():
    df = pd.read_csv("play_tennis.csv")
    X = df.drop("play", axis=1)
    Y = df["play"]
    model = RandomForestClassifier(random_state=8)
    with pytest.raises(ValueError):
        model.predict(X)
    with pytest.raises(ValueError):
        model.predict_proba(X)