
from sklearn.metrics.classification import confusion_matrix
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

import helper as h
import config


def main():
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
    elif page == "Data exploration":
        df = h.load_data()
        st.title("Data Exploration")
        plot_type = st.selectbox("Choose plot type", [
                                 'Box plot', 'Scatter plot', 'Join plot', 'Pair plot', 'Correlation'])
        if plot_type == 'Box plot':
            x_axis = st.selectbox(
                "Choose a variable for the x-axis", selected_feature, index=1)
            y_axis = st.selectbox(
                "Choose a variable for the y-axis", selected_feature, index=2)
            h.visualize_data(df, x_axis, y_axis)
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
        df = h.load_data()
        X = df[config.selected_feature]
        y = df['target']
        split = st.checkbox("Change train test split?")
        feature_selection = st.checkbox("Select features?")
        if split:
            split_size = st.slider("Choose split percentage: ",0.01, 0.99, 0.2)
            X_train, X_test, y_train, y_test = h.train_test(X, y, split_size)
            fig = plt.figure(figsize=(5, 5))
            fig, axes = plt.subplots(nrows=1, ncols=2)
            selected = st.selectbox("selected columns", config.selected_feature)
            sns.distplot(ax=axes[0], x=X_train[selected])
            sns.distplot(ax=axes[1], x=X_test[selected])
            st.pyplot(fig)
        else:
            X_train, X_test, y_train, y_test = h.train_test(X,y,0.2)
        rf = RandomForestClassifier(max_depth=3, random_state=111)

        if feature_selection:
            selected = st.multiselect("selected columns", selected_feature)
            rf.fit(X_train[selected], y_train)
            y_pred = rf.predict(X_test[selected])
            user_input = h.get_user_input(df, selected)
        else:
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)
            user_input = h.get_user_input(df, config.selected_feature)
        st.subheader('Model Test Accuracy Score: '+ str(round(accuracy_score(y_test, y_pred),4)*100)+' %')
        sns.heatmap(confusion_matrix(y_test, y_pred),
                    annot=True, cmap='coolwarm')
        st.pyplot()
        prediction = rf.predict(user_input)
        st.subheader('Cancer Result: ' + ('Benign', 'Malignant')[int(prediction)])


if __name__ == '__main__':
    main()
