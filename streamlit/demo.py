import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
import streamlit.components.v1 as components
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

with st.sidebar:
    selected = option_menu(
        menu_title = None,
        options = ['Presentación', 'Decision Tree', 'Random Forest', 'Documentación']
    )

if selected == 'Presentación':
    # Título de la página
    st.title('Trabajo Prácito Algoritmos - 2024')

    # Descripción del trabajo práctico
    st.write("""Lorem ipsum dolor sit amet, consectetur adipiscing elit. Pellentesque laoreet semper magna, in aliquam ante consequat eget
            Praesent dictum auctor odio ut accumsan.Donec venenatis ante ut enim tempus vestibulum. Aliquam non venenatis dolor, sit amet
            aliquam sapien. Donec eget orci maximus velit ultrices dapibus quis et felis. Etiam vitae nunc cursus ligula lobortis finibus
            et ut metus. Nunc nec faucibus ex, consectetur vestibulum risus.""")


    # Utiliza el componente HTML de Streamlit para mostrar el iframe
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
    st.write('### Play Tennis Dataset')
    df = pd.read_csv('../play_tennis.csv')
    st.write(df.head(10))

    # Mostrar codigo
    st.write('### Code')
    with st.echo():
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
        from sklearn.preprocessing import LabelEncoder

        df = pd.read_csv('../play_tennis.csv')

        X = df.drop("play", axis=1)
        Y = df['play']

        le = LabelEncoder()

        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = le.fit_transform(X[col])

        # Dividir los datos
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        # Entrenar el modelo
        clf = DecisionTreeClassifier()  # CART
        clf.fit(X_train, y_train)

        # Predecir
        y_pred = clf.predict(X_test)                            



    # El usuario puede cambiar los valores de los hiperparámetros (max_depth, min_samples_split, min_samples_leaf, min_impurity_decrease, algorithm)
    max_depth = st.slider('Max Depth', min_value=1, max_value=10, value=3, help='The maximum depth of the tree')
    min_samples_split = st.slider('Min Samples Split', min_value=2, max_value=10, value=2, help='The minimum number of samples required to split an internal node')
    min_samples_leaf = st.slider('Min Samples Leaf', min_value=1, max_value=10, value=1, help='The minimum number of samples required to be at a leaf node')
    min_impurity_decrease = st.slider('Min Impurity Decrease', min_value=0.0, max_value=0.5, value=0.0, help='A node will be split if this split induces a decrease of the impurity greater than or equal to this value')
    algorithm = st.selectbox('Algorithm', ['CART', 'ID3'], help='The algorithm to use when splitting the tree')

        

    # Se muestra el resultado de la suma al presionar el botón "Run"
    if st.button('Run', key='run_button', help='Click to run the code'):
        # Entrenar el modelo
        clf = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, min_impurity_decrease=min_impurity_decrease)  # CART
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
    st.write('### Play Tennis Dataset')
    df = pd.read_csv('../play_tennis.csv')
    st.write(df.head(10))

    # Mostrar codigo
    st.write('### Code')
    with st.echo():
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
        from sklearn.preprocessing import LabelEncoder

        df = pd.read_csv('../play_tennis.csv')

        X = df.drop("play", axis=1)
        Y = df['play']

        le = LabelEncoder()

        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = le.fit_transform(X[col])

        # Dividir los datos
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        # Entrenar el modelo
        clf = RandomForestClassifier()
        clf.fit(X_train, y_train)

        # Predecir
        y_pred = clf.predict(X_test)

    # El usuario puede cambiar los valores de los hiperparámetros (n_estimators, max_depth, min_samples_split, min_samples_leaf, min_impurity_decrease, algorithm)
    n_estimators = st.slider('N Estimators', min_value=1, max_value=100, value=100, help='The number of trees in the forest')
    max_depth = st.slider('Max Depth', min_value=1, max_value=10, value=3, help='The maximum depth of the tree')
    min_samples_split = st.slider('Min Samples Split', min_value=2, max_value=10, value=2, help='The minimum number of samples required to split an internal node')
    min_samples_leaf = st.slider('Min Samples Leaf', min_value=1, max_value=10, value=1, help='The minimum number of samples required to be at a leaf node')
    min_impurity_decrease = st.slider('Min Impurity Decrease', min_value=0.0, max_value=0.5, value=0.0, help='A node will be split if this split induces a decrease of the impurity greater than or equal to this value')
    algorithm = st.selectbox('Algorithm', ['CART', 'ID3'], help='The algorithm to use when splitting the tree')

    # Se muestra el resultado de la suma al presionar el botón "Run"
    if st.button('Run', key='run_button', help='Click to run the code'):
        # Entrenar el modelo
        clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, min_impurity_decrease=min_impurity_decrease)
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

elif selected == 'Documentación':
    with open('/Users/santiagodarnes/Documents/UNSAM/Algoritmos2/Sin título/TPFinalAlgo2/src/docs/_build/html/index.html', 'r', encoding='utf-8') as f:
        html_content = f.read()

    # Muestra el contenido HTML en Streamlit
    components.html(html_content, height=800, scrolling=True)