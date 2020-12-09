from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import learning_curve
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

import helper as h
import config
import pandas as pd


def predictions(df, size, user_input, selected=False):
    X_train, y_train, X_test, y_test, user_input = scaling(selected, df, size, user_input)

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
    return models.sort_values(by='Test Score', ascending=False)


def scaling(selected, df, size, user_input):
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
    return X_train, y_train, X_test, y_test, user_input

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


    '''if method == 'Random Forest':
        pred_train = classifier.predict_proba(X_train)
        pred_test = classifier.predict_proba(X_test)
        fpr, tpr, _ = roc_curve(y_train,  pred_train[:, 1])
        auc = roc_auc_score(y_train, pred_train[:, 1])
        plt.plot(fpr, tpr, label="Train Set, auc="+str(auc))
        fpr, tpr, _ = roc_curve(y_test,  pred_test[:, 1])
        auc = roc_auc_score(y_test, pred_test[:, 1])
        plt.plot(fpr, tpr, label="Test Set, auc="+str(auc))
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.legend(loc=4)
        st.pyplot()'''
    return accuracy, predict

def plot_learning_curve(df, selected=False):
    if selected:
        X = df[selected]
    else:
        X = df[config.selected_feature]
    y = df['target']
    train_sizes, train_scores, test_scores = learning_curve(RandomForestClassifier(max_depth=3, random_state=111),
                                                            X,
                                                            y,
                                                            cv=5,
                                                            scoring='accuracy',
                                                            n_jobs=-1,
                                                            train_sizes=np.linspace(0.01, 1.0, 20))


    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)

    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)


    fig, axes = plt.subplots()
    plt.plot(train_sizes, train_mean, '--',
            color="#111111",  label="Training score")
    plt.plot(train_sizes, test_mean, color="#111111",
            label="Test score")

    plt.fill_between(train_sizes, train_mean - train_std,
                    train_mean + train_std, color="#DDDDDD")
    plt.fill_between(train_sizes, test_mean - test_std,
                    test_mean + test_std, color="#DDDDDD")

    plt.title("Learning Curve")
    plt.xlabel("Training Set Size"), plt.ylabel(
        "Accuracy Score"), plt.legend(loc="best")
    st.pyplot(fig)
