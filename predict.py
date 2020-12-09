from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics.classification import confusion_matrix
from sklearn.preprocessing import StandardScaler

import helper as h
import config
import pandas as pd


def predictions(df, size, user_input, selected=False):
    if selected:
        X = df[selected]
    else:
        X = df[config.selected_feature]
    y = df['target']
    X_train, X_test, y_train, y_test = h.train_test(X, y, size)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    user_input = scaler.transform(user_input)

    accuracy_regression, predict_regression = fit_predict(
        'Logistic', X_train, y_train, X_test, y_test, user_input)
    accuracy_KNN, predict_KNN = fit_predict(
        'KNN', X_train, y_train, X_test, y_test, user_input)
    accuracy_decision_tree, predict_decision_tree = fit_predict(
        'Decision Tree', X_train, y_train, X_test, y_test, user_input)
    accuracy_rf, predict_rf = fit_predict(
        'Random Forest', X_train, y_train, X_test, y_test, user_input)

    models = pd.DataFrame({
        'Model': ['KNN (k=5)', 'Logistic Regression', 'Decision Tree','Random Forest'],
        'Test Score': [accuracy_KNN, accuracy_regression, accuracy_decision_tree,accuracy_rf],
        'New Data Prediction': [predict_KNN, predict_regression, predict_decision_tree, predict_rf]})
    return models.sort_values(by='Test Score', ascending=False, ignore_index=True)

def fit_predict(method, X_train, y_train, X_test, y_test, user_input):
    if method == 'Logistic':
        classifier = LogisticRegression(random_state=111)
    elif method == 'KNN':
        classifier = KNeighborsClassifier(
            n_neighbors=5, metric='minkowski', p=2)
    elif method=='Decision Tree':
        classifier = DecisionTreeClassifier(max_depth=3, random_state=111)
    elif method == 'Random Forest':
        classifier = RandomForestClassifier(max_depth=3, random_state=111)
    classifier.fit(X_train, y_train)
    y_predict = classifier.predict(X_test)
    accuracy = round(accuracy_score(y_test, y_predict) * 100, 2)
    prediction = classifier.predict(user_input)
    predict = ('Benign', 'Malignant')[int(prediction)]
    #fig, axes = plt.subplots(nrows=1, ncols=1)
    #sns.heatmap(confusion_matrix(y_test, y_predict),
    #           annot=True, cmap='coolwarm')
    #st.pyplot(fig)
    return accuracy, predict

