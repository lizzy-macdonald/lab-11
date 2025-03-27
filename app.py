import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
import pandas as pd

# Load the trained model and preprocessing objects
model = tf.keras.models.load_model("tf_bridge_model.h5")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("encoder.pkl")

# Streamlit UI
st.title("Bridge Max Load Prediction App")
st.write("Enter bridge details to predict the maximum load capacity in tons.")

# User Inputs
span_ft = st.number_input("Span (ft):", min_value=10, max_value=1000, value=100)
deck_width_ft = st.number_input("Deck Width (ft):", min_value=5, max_value=200, value=20)
age_years = st.number_input("Age (Years):", min_value=0, max_value=150, value=30)
num_lanes = st.number_input("Number of Lanes:", min_value=1, max_value=10, value=2)
condition_rating = st.slider("Condition Rating (1=Poor, 5=Excellent):", min_value=1, max_value=5, value=3)

# Material Selection
material_options = encoder.categories_[0]
material = st.selectbox("Bridge Material:", material_options)

# Preprocessing user input
material_encoded = encoder.transform([[material]])
user_input = np.array([[span_ft, deck_width_ft, age_years, num_lanes, condition_rating]])
user_input_scaled = scaler.transform(user_input)
user_input_final = np.hstack((user_input_scaled, material_encoded))

# Predict Button
if st.button("Predict Max Load"):
    prediction = model.predict(user_input_final)
    st.success(f"Predicted Maximum Load Capacity: {prediction[0][0]:.2f} tons")
