import streamlit as st
import pandas as pd
import joblib
from tensorflow import keras
import numpy as np

# === Load pre-trained models and preprocessors ===
model_selected = keras.models.load_model('model_selected.h5')
model_all = keras.models.load_model('model_all.h5')
preprocessor_selected = joblib.load('preprocessor_selected.pkl')
preprocessor_all = joblib.load('preprocessor_all.pkl')

st.title("Lab 11 Bridge Data")

# Sidebar: let the user choose which model to use
model_choice = st.sidebar.radio("Select Model", ("Essential Features Model", "All Features Model"))
st.header("Input Bridge Data (Essential Only)")

# User inputs for essential features
Age = st.number_input("Age", min_value=0, max_value=100, value=30)
Span_ft = st.number_input("Span ft", min_value=100, max_value=600, value=300)
Deck_Width_ft = st.number_input("Deck width ft", min_value=20, max_value=60, value=50)
Condition_Rating = st.number_input("Deck Rating (1-5)", min_value=1, max_value=10, value=4)
Num_Lanes = st.number_input("Num Lanes", min_value=1, max_value=6, value=6)
Material = st.selectbox("Material", options=["Steel", "Composite", "Concrete"])

# Convert Material to match expected categories
expected_materials = preprocessor_selected.named_transformers_['Material'].categories_[0]  # Extract expected categories

if Material not in expected_materials:
    st.error(f"Unexpected material: {Material}. Expected: {expected_materials}")
    st.stop()

# Ensure Material is encoded correctly
input_data = pd.DataFrame({
    'Age': [Age],
    'Span ft': [Span_ft],
    'Deck width ft': [Deck_Width_ft],
    'Condition Rating': [Condition_Rating],
    'Num Lanes': [Num_Lanes],
    'Material': [Material]  # Keep as string; let preprocessor handle encoding
})

# Handle missing columns
expected_columns = preprocessor_selected.feature_names_in_
input_data = input_data.reindex(columns=expected_columns, fill_value=np.nan)  # Fill missing values with NaN

# Debugging output
st.write("Final Input Data:", input_data)

# When the user clicks the Predict button
if st.button("Predict Max Load Tons"):
    try:
        processed_data = preprocessor_selected.transform(input_data)
        prediction = model_selected.predict(processed_data)
        st.success(f"Predicted Max Load Tons (Essential Model): {prediction[0][0]:,.2f}")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
