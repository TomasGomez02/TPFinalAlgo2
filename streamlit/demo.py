import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
import streamlit.components.v1 as components
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

with st.sidebar:
    selected = option_menu(
        menu_title = None,
        options = ['Presentación', 'Decision Tree', 'Random Forest']
    )

if selected == 'Presentación':
    st.title('Trabajo Final Algoritmos 2 - 2024')

    components.html(
        f"""
        <div style="position: relative; width: 100%; height: 0; padding-top: 56.2500%;
    padding-bottom: 0; box-shadow: 0 2px 8px 0 rgba(63,69,81,0.16); margin-top: 1.6em; margin-bottom: 0.9em; overflow: hidden;
    border-radius: 8px; will-change: transform;">
    <iframe loading="lazy" style="position: absolute; width: 100%; height: 100%; top: 0; left: 0; border: none; padding: 0;margin: 0;"
        src="https:&#x2F;&#x2F;www.canva.com&#x2F;design&#x2F;DAGH32wWfR4&#x2F;STBuI6DGsnYCBCu5wimUNA&#x2F;view?embed" allowfullscreen="allowfullscreen" allow="fullscreen">
    </iframe>
    </div>
    <a href="https:&#x2F;&#x2F;www.canva.com&#x2F;design&#x2F;DAGH32wWfR4&#x2F;STBuI6DGsnYCBCu5wimUNA&#x2F;view?utm_content=DAGH32wWfR4&amp;utm_campaign=designshare&amp;utm_medium=embeds&amp;utm_source=link" target="_blank" rel="noopener">TP ALGORITMOS</a>
        """,
        height=450
    )

    

elif selected == 'Decision Tree':
    # Mostrar dataset
    st.write('## Decision Tree')
    st.write('### Star Classification Dataset')
    df = pd.read_csv('../star_classification.csv').drop(["obj_ID","run_ID","rerun_ID", "field_ID","spec_obj_ID", "fiber_ID"], axis=1)
    st.write(df.head(10))

    st.write('---')

    # Tree Models
    st.write('### Decision Tree - Tree Models')
    with st.echo():
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from treeModels import DecisionTreeClassifier
        from treeModels.decision_algorithms import DecisionAlgorithm
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        from plot.plot_tree import PlotTree

        df = pd.read_csv('../star_classification.csv')
        df = df.assign(quality = [str(x) for x in df['cam_col']]).sample(1000)

        X = df.drop("class", axis=1)
        Y = df['class']

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        clf = DecisionTreeClassifier()
                                   



    max_depth_1 = st.slider('Max Depth', min_value=1, max_value=10, value=4, help='The maximum depth of the tree')
    min_samples_split_1 = st.slider('Min Samples Split', min_value=2, max_value=10, value=2, help='The minimum number of samples required to split an internal node')
    min_samples_leaf_1 = st.slider('Min Samples Leaf', min_value=1, max_value=10, value=1, help='The minimum number of samples required to be at a leaf node')
    min_impurity_decrease_1 = st.slider('Min Impurity Decrease', min_value=0.0, max_value=0.5, value=0.0, help='A node will be split if this split induces a decrease of the impurity greater than or equal to this value')    
    algorithm_1 = st.selectbox('Algorithm', ['C45', 'ID3'], help='The algorithm to use when splitting the tree')
    
    if st.button('Run', key='run_button', help='Click to run the code'):
        
        if algorithm_1 == 'C45':
            algorithm_1 = DecisionAlgorithm.C45
        else:
            algorithm_1 = DecisionAlgorithm.ID3

        clf = DecisionTreeClassifier(max_depth=max_depth_1, min_samples_split=min_samples_split_1, min_samples_leaf=min_samples_leaf_1, min_impurity_decrease=min_impurity_decrease_1, algorithm=algorithm_1)
        clf.fit(X_train, y_train)

        # Predecir
        y_pred = clf.predict(X_test)

        # Calcular métricas
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')

        st.write('Accuracy:', round(accuracy, 3))
        st.write('Precision:', round(precision, 3))
        st.write('Recall:', round(recall, 3))
        st.write('F1:', round(f1, 3))
        
        PlotTree(clf.tree).show()


    st.write('---')

    # SKLearn
    st.write('### SKLearn')
    with st.echo():
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.tree import DecisionTreeClassifier as DecisionTreeClassifierSK
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        df = pd.read_csv('../star_classification.csv')
        df.assign(quality = [str(x) for x in df['cam_col']]).sample(1000)

        X = df.drop("class", axis=1)
        Y = df['class']

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        clf = DecisionTreeClassifierSK()
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)                            



    max_depth = st.slider('Max Depth 2', min_value=1, max_value=10, value=4, help='The maximum depth of the tree')
    min_samples_split = st.slider('Min Samples Split 2', min_value=2, max_value=10, value=2, help='The minimum number of samples required to split an internal node')
    min_samples_leaf = st.slider('Min Samples Leaf 2', min_value=1, max_value=10, value=1, help='The minimum number of samples required to be at a leaf node')
    min_impurity_decrease = st.slider('Min Impurity Decrease 2', min_value=0.0, max_value=0.5, value=0.0, help='A node will be split if this split induces a decrease of the impurity greater than or equal to this value')    

    
    if st.button('Run', key='run_button_sk', help='Click to run the code'):
        # Entrenar el modelo
        clf = DecisionTreeClassifierSK(max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, min_impurity_decrease=min_impurity_decrease)  # CART
        clf.fit(X_train, y_train)

        # Predecir
        y_pred = clf.predict(X_test)

        # Calcular métricas
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')

        st.write('Accuracy:', accuracy)
        st.write('Precision:', precision)
        st.write('Recall:', recall)
        st.write('F1:', f1)

        # Mostrar el arbol de decisión
        plt.figure(figsize=(20, 10))
        plot_tree(clf, filled=True, feature_names=X.columns)

        plt.savefig('decision_tree.png')
        st.image('decision_tree.png')
        plt.close()  

elif selected == 'Random Forest':
    st.write('## Random Forest')
    st.write('### Star Classification Dataset')
    df = pd.read_csv('../star_classification.csv').drop(["obj_ID","run_ID","rerun_ID", "field_ID","spec_obj_ID", "fiber_ID"], axis=1)
    st.write(df.head(10))

    st.write('---')

    # Tree Models
    st.write('### Random Forest - Tree Models')
    with st.echo():
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from treeModels import RandomForestClassifier
        from treeModels.decision_algorithms import DecisionAlgorithm
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        df = pd.read_csv('../star_classification.csv')
        df = df.assign(quality = [str(x) for x in df['cam_col']]).sample(1000)

        X = df.drop("class", axis=1)
        Y = df['class']

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        clf = RandomForestClassifier()
                                   

    n_estimators_1 = st.slider('N Estimators 1', min_value=1, max_value=100, value=10, help='The number of trees in the forest')
    max_features_1 = st.selectbox('Max Features 1', [ 'None', 'sqrt', 'log2'], help='The number of features to consider when looking for the best split')
    max_depth_1 = st.slider('Max Depth 1', min_value=1, max_value=10, value=4, help='The maximum depth of the tree')
    min_samples_split_1 = st.slider('Min Samples Split 1', min_value=2, max_value=10, value=2, help='The minimum number of samples required to split an internal node')
    min_samples_leaf_1 = st.slider('Min Samples Leaf 1', min_value=1, max_value=10, value=1, help='The minimum number of samples required to be at a leaf node')
    min_impurity_decrease_1 = st.slider('Min Impurity Decrease 1', min_value=0.0, max_value=0.5, value=0.0, help='A node will be split if this split induces a decrease of the impurity greater than or equal to this value')
    algorithm_1 = st.selectbox('Algorithm', ['C45', 'ID3'], help='The algorithm to use when splitting the tree')
    
    if st.button('Run', key='run_button_2', help='Click to run the code'):

        if max_features_1 == 'None':
            max_features_1 = None
        
        if algorithm_1 == 'C45':
            algorithm_1 = DecisionAlgorithm.C45
        else:
            algorithm_1 = DecisionAlgorithm.ID3

        clf = RandomForestClassifier(n_estimators=n_estimators_1, max_features=max_features_1, max_depth=max_depth_1, min_samples_split=min_samples_split_1, min_samples_leaf=min_samples_leaf_1, min_impurity_decrease=min_impurity_decrease_1, algorithm=algorithm_1)
        clf.fit(X_train, y_train)

        # Predecir
        y_pred = clf.predict(X_test)

        # Calcular métricas
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')

        st.write('Accuracy:', round(accuracy, 3))
        st.write('Precision:', round(precision, 3))
        st.write('Recall:', round(recall, 3))
        st.write('F1:', round(f1, 3))

    st.write('---')

    st.write('### SKLearn')
    with st.echo():
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        df = pd.read_csv('../star_classification.csv')
        df = df.assign(quality = [str(x) for x in df['cam_col']]).sample(1000)

        X = df.drop("class", axis=1)
        Y = df['class']

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        clf = RandomForestClassifier()
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

    
    n_estimators = st.slider('N Estimators', min_value=1, max_value=100, value=100, help='The number of trees in the forest')
    max_feature = st.selectbox('Max Features', ['sqrt', 'log2', 'None'], help='The number of features to consider when looking for the best split')
    max_depth = st.slider('Max Depth', min_value=1, max_value=10, value=4, help='The maximum depth of the tree')
    min_samples_split = st.slider('Min Samples Split', min_value=2, max_value=10, value=2, help='The minimum number of samples required to split an internal node')
    min_samples_leaf = st.slider('Min Samples Leaf', min_value=1, max_value=10, value=1, help='The minimum number of samples required to be at a leaf node')
    min_impurity_decrease = st.slider('Min Impurity Decrease', min_value=0.0, max_value=0.5, value=0.0, help='A node will be split if this split induces a decrease of the impurity greater than or equal to this value')

    
    if st.button('Run', key='run_button_3', help='Click to run the code'):
        
        if max_feature == 'None':
            max_feature = None

        clf = RandomForestClassifier(n_estimators=n_estimators, max_features=max_feature, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, min_impurity_decrease=min_impurity_decrease)
        clf.fit(X_train, y_train)

        # Predecir
        y_pred = clf.predict(X_test)

        # Calcular métricas
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')

        st.write('Accuracy:', accuracy)
        st.write('Precision:', precision)
        st.write('Recall:', recall)
        st.write('F1:', f1)

        # Mostrar el arbol de decisión
        plt.figure(figsize=(20, 10))
        plot_tree(clf.estimators_[0], filled=True, feature_names=X.columns)

        plt.savefig('random_forest.png')
        st.image('random_forest.png')
        plt.close()