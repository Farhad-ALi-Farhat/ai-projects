import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression

# Train simple model
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

model = LinearRegression()
model.fit(X, y)

st.title("📈 Linear Regression Predictor")

value = st.number_input("Enter a number:")

if st.button("Predict"):
    prediction = model.predict([[value]])
    st.success(f"Prediction: {prediction[0]:.2f}")
