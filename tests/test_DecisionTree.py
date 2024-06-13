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
    assert model.get_n_leaves() == 11
    assert model.predict_proba(X).shape == (19, 2)
    assert TreeClassifierTM().fit(X.iloc[0:1,:], Y.iloc[0:1]).score(X.iloc[0:1,:], Y.iloc[0:1]) == 1
    with pytest.raises(NotImplementedError):
        model.decision_path(X)

# def test_comparison():
#     df_tm = pd.read_csv("CarEval.csv")
#     df_tm["doors"] = str(df_tm["doors"])
#     df_tm["persons"] = str(df_tm["persons"])
#     df_skl = df_tm.copy(True)
#     index = 1500
    
#     for column in df_skl:
#         if df_skl[column].dtype == object:
#             df_skl[column] = LabelEncoder().fit_transform(df_skl[column])
    
#     X_tm, Y_tm = df_tm.drop("class values", axis=1), df_tm["class values"]
#     X_skl, Y_skl = df_skl.drop("class values", axis=1), df_skl["class values"]

def test_params(X, Y):
    assert TreeClassifierTM(max_depth=3).fit(X, Y).get_depth() == 3
    assert len(TreeClassifierTM(min_samples_split=5).fit(X, Y).tree.samples) >= 5
    params = {"max_depth":200, "min_samples_split":9, "min_samples_leaf":10,
    "min_impurity_decrease": 1, "algorithm":"C4.5"}
    assert TreeClassifierTM().set_params(**params).get_params() == params

@pytest.mark.xfail
def test_C4_5():
    # Hay que completar este test
    df = pd.read_csv("CarEval.csv")
    X = df.drop("class values", axis=1)
    Y = df["class values"]
    model = TreeClassifierTM(algorithm="C4.5").fit(X.iloc[5:,:], Y.iloc[5:])
    model.predict(X.iloc[0:5,:])
    

def test_Exceptions_Errors(X, Y):
    overFittedTree = TreeClassifierTM()
    with pytest.raises(ValueError):
        overFittedTree.predict(X)
    overFittedTree.set_params(nombre="gonza")
    with pytest.raises(AttributeError):
        overFittedTree.nombre