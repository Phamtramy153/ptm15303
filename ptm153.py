import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

# Load dữ liệu Iris
iris = load_iris()
X = iris.data
y = iris.target

# Tạo một dataframe
df = pd.DataFrame(X, columns=iris.feature_names)

# Tiêu đề ứng dụng
st.title('Iris Flower Species Classifier')

# Hiển thị dataframe trên ứng dụng
st.write("Here are the first five rows of the Iris dataset:")
st.write(df.head())

# Cho phép người dùng nhập các thông số của hoa Iris
st.sidebar.header('User Input Parameters')

def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

user_input = user_input_features()

# Hiển thị thông số người dùng nhập
st.subheader('User Input parameters')
st.write(user_input)

# K-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)
prediction = knn.predict(user_input)

st.subheader('Prediction')
st.write(iris.target_names[prediction])
st.write('Probabilities')
st.write(knn.predict_proba(user_input))
