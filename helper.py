import pandas as pd
import numpy as np
import altair as alt
import math
import streamlit as st
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

@st.cache
def load_data():
    cancer = load_breast_cancer()
    df = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns=np.append(
        cancer['feature_names'], 'target'))
    return df


def visualize_data(df, x_axis, y_axis):
    graph = alt.Chart(df).mark_circle(size=60).encode(
        x=x_axis, y=y_axis).interactive()
    st.write(graph)


def get_user_input(df, selected_feature):
    user_data = {feature: st.sidebar.slider(feature, math.floor(df[feature].min())*1.0, math.floor(
        df[feature].max())*1.0+1, math.floor(df[feature].mean())*1.0) for feature in selected_feature}
    features = pd.DataFrame(user_data, index=[0])
    return features


def train_test(X, y, size):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=size, random_state=111)
    return X_train, X_test, y_train, y_test
