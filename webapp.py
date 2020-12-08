import pandas as pd
import numpy as np
from scipy.sparse import data
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt
import math
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer


def main():
    selected_feature =['area error',
    'compactness error',
    'mean area',
    'mean compactness',
    'mean concave points',
    'mean concavity',
    'mean perimeter',
    'mean radius',
    'mean smoothness',
    'mean texture',
    'perimeter error',
    'radius error',
    'worst compactness',
    'worst concavity',
    'worst perimeter',
    'worst radius'
    ]
    page = st.sidebar.selectbox("Choose a page", ["DataFrame", "Data exploration", "Prediction"])
    
    if page == "DataFrame":
        st.header("This is your data explorer.")
        df = load_data()
        show_selected = st.checkbox("Show selected data?")
        if show_selected is not True:
            st.write(df.head(20))
        else:
            number = st.slider("Choose the interval: ", 0, df.shape[0], (0, 19))
            selected_columns = st.multiselect("selected columns",df.columns)
            if number[0] <= number[1] and len(selected_columns) > 0:
                st.write(df[selected_columns][number[0]:number[1]+1])
        st.header("Data information")
        selected_columns = st.multiselect("selected columns", df.columns)
        if len(selected_columns) > 0:
            st.write(df[selected_columns].describe().T)
    elif page == "Data exploration":
        df = load_data()
        st.title("Data Exploration")
        plot_type = st.selectbox("Choose plot type", [
                                 'Box plot', 'Scatter plot', 'Join plot', 'Pair plot', 'Correlation'])
        if plot_type == 'Box plot':
            x_axis = st.selectbox(
                "Choose a variable for the x-axis", selected_feature, index=1)
            y_axis = st.selectbox(
                "Choose a variable for the y-axis", selected_feature, index=2)
            visualize_data(df, x_axis, y_axis)
        elif plot_type == 'Scatter plot':
             x_axis = st.selectbox(
                "Choose a variable for the x-axis", selected_feature, index=1)
             y_axis = st.selectbox(
                 "Choose a variable for the y-axis", selected_feature, index=2)
             sns.scatterplot(x=x_axis, y=y_axis, data=df)
             hue = st.checkbox("with hue")
             if hue:
                sns.scatterplot(x=x_axis, y=y_axis, hue='target', data=df)
             st.pyplot()
        elif plot_type == 'Join plot':
            x_axis = st.selectbox(
                "Choose a variable for the x-axis", selected_feature, index=1)
            y_axis = st.selectbox(
                "Choose a variable for the y-axis", selected_feature, index=2)
            sns.jointplot(x=x_axis, y=y_axis, kind='hex', data=df)
            st.pyplot()
        elif plot_type == 'Pair plot':
            selected_columns = st.multiselect("selected columns", df.columns)
            if len(selected_columns) > 1:
                sns.pairplot(df[selected_columns])
                st.pyplot()
        elif plot_type == 'Correlation':
            selected_columns = st.multiselect("selected columns", df.columns)
            if len(selected_columns) > 1:
                sns.heatmap(df[selected_columns].corr(), annot=True, cmap='coolwarm')
                st.pyplot()
    elif page == "Prediction":
        st.write(""" # Breast Cancer Prediction """)
        df = load_data()
        X = df[selected_feature]
        y = df['target']
        split = st.checkbox("change train test split?")
        if split:
            size = st.slider("Choose split percentage: ",0.01, 0.99, 0.2)
            X_train, X_test, y_train, y_test = train_test(X, y, size)
            fig = plt.figure(figsize=(5, 5))
            fig, axes = plt.subplots(nrows=1, ncols=2)
            selected_column = st.selectbox("selected columns", selected_feature)
            sns.distplot(ax=axes[0], x=X_train[selected_column])
            sns.distplot(x=X_test[selected_column], ax=axes[1])
            st.pyplot(fig)

        else:
            X_train, X_test, y_train, y_test = train_test(X,y,0.2)
        rf = RandomForestClassifier()
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        st.subheader('Model Test Accuracy Score: '+ str(round(accuracy_score(y_test, y_pred),4)*100)+' %')


        user_input = get_user_input(df, selected_feature)
        prediction = rf.predict(user_input)
        st.subheader('Cancer Result: ' + ('Benign', 'Malignant')[int(prediction)])


@st.cache
def load_data():
    cancer = load_breast_cancer()
    df = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns=np.append(cancer['feature_names'], 'target'))
    return df


def visualize_data(df, x_axis, y_axis):
    graph = alt.Chart(df).mark_circle(size=60).encode(x=x_axis,y=y_axis).interactive()
    st.write(graph)

def get_user_input(df, selected_feature):
    user_data = {feature: st.sidebar.slider(feature, math.floor(df[feature].min())*1.0, math.floor(
        df[feature].max())*1.0+1, math.floor(df[feature].mean())*1.0) for feature in selected_feature}
    features = pd.DataFrame(user_data, index=[0])
    return features


def train_test(X,y, size):
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=size, random_state=111)
    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    main()
