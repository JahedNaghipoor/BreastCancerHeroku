
from seaborn import distributions
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

import helper as h
import config
import predict


def main():
    
    h.max_width()
    #st.set_option('deprecation.showPyplotGlobalUse', False)
    
    selected_feature = config.selected_feature
    
    page = st.sidebar.selectbox("Choose a page", ["DataFrame", "Data exploration", "Prediction"])
    
    if page == "DataFrame":
        st.header("Dataframe")
        df = h.load_data()
        show_selected = st.checkbox("Show selected data?")
        if show_selected is not True:
            st.write(df.head(20))
        else:
            number = st.slider("Choose the interval: ", 0, df.shape[0], (0, 19))
            selected_columns = st.multiselect("selected columns",df.columns)
            if number[0] <= number[1] and len(selected_columns) > 0:
                st.write(df[selected_columns][number[0]:number[1]+1])
        extra_info = st.checkbox("Show extra information?")
        if extra_info:
            st.write(df.describe().T)
    elif page == "Data exploration":
        df = h.load_data()
        st.title("Data Exploration")
        plot_type = st.selectbox("Choose plot type", [
                                 'Scatter plot', 'Dist plot', 'Join plot', 'Pair plot', 'Correlation', 'Train-test dist plot'])
        if plot_type == 'Scatter plot':
             x_axis = st.selectbox(
                "Choose a variable for the x-axis", selected_feature, index=1)
             y_axis = st.selectbox(
                 "Choose a variable for the y-axis", selected_feature, index=2)
             sns.scatterplot(x=x_axis, y=y_axis, data=df)
             hue = st.checkbox("with hue")
             if hue:
                sns.scatterplot(x=x_axis, y=y_axis, hue='target', data=df)
             st.pyplot()
        elif plot_type == 'Dist plot':
            x_axis = st.selectbox(
                "Choose a variable for the x-axis", selected_feature, index=0)
            sns.distplot(x=df[x_axis], bins=30)
            st.pyplot()
        elif plot_type == 'Join plot':
            x_axis = st.selectbox(
                "Choose a variable for the x-axis", selected_feature, index=0)
            y_axis = st.selectbox(
                "Choose a variable for the y-axis", selected_feature, index=1)
            sns.jointplot(x=x_axis, y=y_axis, kind='hex', data=df)
            st.pyplot()
        elif plot_type == 'Pair plot':
            selected_columns = st.multiselect("selected columns", df.columns.drop('target'))
            if len(selected_columns) > 1:
                sns.pairplot(df[selected_columns])
                st.pyplot()
        elif plot_type == 'Correlation':
            selected_columns = st.multiselect("selected columns", df.columns)
            if len(selected_columns) > 1:
                fig, axes = plt.subplots()
                sns.heatmap(df[selected_columns].corr(), annot=True, cmap='coolwarm')
                st.pyplot(fig)
        elif plot_type == 'Train-test dist plot':
            split_size = st.slider(
                "Choose split percentage: ", 0.01, 0.99, 0.2)
            X_train, X_test, y_train, y_test = h.train_test(
                df[config.selected_feature], df['target'], split_size)
            selected = st.selectbox("selected columns", config.selected_feature)
            fig, axes = plt.subplots(nrows=1, ncols=2)
            sns.distplot(ax=axes[0], x=X_train[selected])
            sns.distplot(ax=axes[1], x=X_test[selected])
            st.pyplot(fig)
    elif page == "Prediction":
        st.write(""" # Breast Cancer Prediction """)
        df = h.load_data()
        X = df[config.selected_feature]
        y = df['target']
        split_size = 0.2
        split = st.checkbox("Change train test split?")
        feature_selection = st.checkbox("Select features?")
        learning_curve = st.checkbox("Show learning curve for Random Forest?")
        if split:
            split_size = st.slider("Choose split percentage: ",0.01, 0.99, 0.2)
            X_train, X_test, y_train, y_test = h.train_test(X, y, split_size)
        if feature_selection or learning_curve:
            selected = st.multiselect("Select some columns", selected_feature)
            if len(selected) > 0 :
                user_input = h.get_user_input(df, selected)
                st.write(predict.predictions(df, split_size, user_input, selected))
            if learning_curve:
                predict.plot_learning_curve(df, selected)
        else:
            user_input = h.get_user_input(df, config.selected_feature)
            st.write(predict.predictions(df, split_size, user_input))

if __name__ == '__main__':
    main()
