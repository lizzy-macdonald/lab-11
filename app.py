import streamlit as st
import pandas as pd
import joblib
from tensorflow import keras
import tensorflow as tf

# === Load pre-trained models and preprocessors ===
model_selected = keras.models.load_model('model_selected.h5')
model_all = keras.models.load_model('model_all.h5')
preprocessor_selected = joblib.load('preprocessor_selected.pkl')
preprocessor_all = joblib.load('preprocessor_all.pkl')

# === Define default values for non-essential features for the full model ===
default_values = {
  'Age': 30,  
  'Span ft': 300,  
  'Deck width ft': 50,  
  'Condition Rating': 4,
  'Num Lanes': 6,  
  'Material': 'Steel',
}

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


# When the user clicks the Predict button
if st.button("Predict Max Load Tons"):
    if model_choice == "Essential Features Model":
        # Build a DataFrame from the essential features only
        input_data = pd.DataFrame({
            'Age': [Age],
            'Span ft': [Span_ft],
            'Deck width ft': [Deck_Width_ft],
            'Condition Rating': [Condition_Rating],
            'Num Lanes': [Num_Lanes],
            'Material': [Material]
        })
        # Preprocess input using the selected-features preprocessor
        processed_data = preprocessor_selected.transform(input_data)
        # Get prediction from the essential-features model
        prediction = model_selected.predict(processed_data)
        st.success(f"Predicted Max Load Tons (Essential Model): ${prediction[0][0]:,.2f}")
    else:
        default_all = pd.read_csv('default_all_features.csv', index_col=0)
        # Now, 'default_all' contains all the features expected by the preprocessor.
        # Overwrite the essential features with user inputs
        default_all.loc[0, 'Age'] = Age_Years
        default_all.loc[0, 'Span ft'] = Span_ft
        default_all.loc[0, 'Deck width ft'] = Deck_Width_ft
        default_all.loc[0, 'Condition Rating'] = Condition_Rating
        default_all.loc[0, 'Num Lanes'] = Num_Lanes
        default_all.loc[0, 'Material'] = Material
        
        processed_data = preprocessor_all.transform(default_all)
        prediction = model_all.predict(processed_data)
        st.success(f"Predicted Max Load Tons (All Features Model): ${prediction[0][0]:,.2f}")
