import streamlit as st
import pandas as pd
import joblib
from tensorflow import keras

# === Load models and preprocessors ===
model_selected = keras.models.load_model('model_selected.h5')
model_all = keras.models.load_model('model_all.h5')
preprocessor_selected = joblib.load('preprocessor_selected.pkl')
preprocessor_all = joblib.load('preprocessor_all.pkl')

st.title("Lab 11 Bridge Data")

# Sidebar: Select model
model_choice = st.sidebar.radio("Select Model", ("Essential Features Model", "All Features Model"))
st.write(f"Selected Model: {model_choice}")  # Debugging: Check selected model

st.header("Input Bridge Data")

# User inputs
Age = st.number_input("Age", min_value=0, max_value=100, value=30)
Span_ft = st.number_input("Span ft", min_value=100, max_value=600, value=300)
Deck_Width_ft = st.number_input("Deck width ft", min_value=20, max_value=60, value=50)
Condition_rating = st.number_input("Deck Rating (1-5)", min_value=1, max_value=10, value=4)
Num_Lanes = st.number_input("Num Lanes", min_value=1, max_value=6, value=6)
Material = st.selectbox("Material", options=["Steel", "Composite", "Concrete"])

if st.button("Predict Max Load Tons"):
    if model_choice == "Essential Features Model":
        # Create input DataFrame
        input_data = pd.DataFrame({
            'Age': [Age],
            'Span ft': [Span_ft],
            'Deck width ft': [Deck_Width_ft],
            'Condition Rating': [Condition_rating],
            'Num Lanes': [Num_Lanes],
            'Material': [Material]
        })

        st.write("Expected Columns:", preprocessor_selected.feature_names_in_)
        st.write("Input Data Columns:", input_data.columns)  # Now inside the block

        try:
            processed_data = preprocessor_selected.transform(input_data)
            prediction = model_selected.predict(processed_data)
            st.success(f"Predicted Max Load Tons: {prediction[0][0]:,.2f}")
        except Exception as e:
            st.error(f"Error in transformation or prediction: {e}")

    else:
        st.error("All Features Model not implemented yet.")
