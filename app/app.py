# app/app.py
import streamlit as st
import requests
import numpy as np
import json

# Set the API URL (BentoML API)
api_url = "http://127.0.0.1:3000/predict"

# Streamlit UI
st.title("Iris Flower Prediction")
st.write("This app predicts the species of Iris flowers based on input features.")

# Input fields for user
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0)

# Prepare input for prediction
input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

if st.button("Predict"):
    # Send input data to BentoML API
    response = requests.post(api_url, json=input_data.tolist())
    if response.status_code == 200:
        result = response.json()
        st.write(f"Prediction: {result['prediction']}")
    else:
        st.write("Error in prediction.")
