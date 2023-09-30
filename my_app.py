import numpy as np
import pandas as pd
import streamlit as st
from lazypredict.Supervised import LazyClassifier
from lazypredict.Supervised import LazyRegressor
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

st.set_page_config(
    page_title="dk - LazyPredict AutoML",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown('## Classification')
with st.spinner('Preparing...'):
    breast_cancer = datasets.load_breast_cancer()
    df_breast_cancer = pd.DataFrame(data=breast_cancer.data, columns=breast_cancer.feature_names)
    X = breast_cancer.data
    y = breast_cancer.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=21)

    clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
    clf_models, clf_predictions = clf.fit(X_train, X_test, y_train, y_test)

    st.markdown('#### Sklearn Breast Cancer Dataset `head()`')
    st.dataframe(df_breast_cancer.head(), use_container_width=True)
    st.markdown('#### `from lazypredict.Supervised import LazyClassifier`')
    with st.expander("expand to see details..."):
        st.write(clf)
    st.markdown('##### Model Evaluation')
    st.dataframe(clf_models, use_container_width=True)

st.markdown('## Regression')
with st.spinner('Preparing...'):
    diabetes = datasets.load_diabetes()
    df_diabetes = pd.DataFrame(data=diabetes.data, columns=diabetes.feature_names)
    X, y = shuffle(diabetes.data, diabetes.target, random_state=21)
    X = X.astype(np.float32)

    offset = int(X.shape[0] * 0.8)

    X_train, y_train = X[:offset], y[:offset]
    X_test, y_test = X[offset:], y[offset:]

    reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
    reg_models, reg_predictions = reg.fit(X_train, X_test, y_train, y_test)

    st.markdown('#### Sklearn Diabetes Dataset `head()`')
    st.dataframe(df_diabetes.head(), use_container_width=True)
    st.markdown('#### `from lazypredict.Supervised import LazyRegressor`')
    with st.expander("expand to see details..."):
        st.write(reg)
    st.markdown('##### Model Evaluation')
    st.dataframe(reg_models, use_container_width=True)
