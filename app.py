import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the trained pipeline
try:
    full_pipeline = joblib.load('full_pipeline.pkl')
except FileNotFoundError:
    st.error("Pipeline file not found. Make sure 'full_pipeline.pkl' is in the same directory.")
    st.stop()

# --- Website Title and Description ---
st.title("Crop Production and Yield Prediction")
st.markdown("""
Welcome to the Crop Production and Yield Prediction App!
Input the details below to get predictions for crop production and yield.
""")

# --- Input Fields ---
st.header("Input Features")

# Get unique values for dropdowns from the original dataset (X_mod)
# Assuming X_mod is available in the environment or loaded from a file
# If X_mod is not available, you would need to load it or get the unique values
# from the original data source. For this example, we'll assume X_mod is a DataFrame.

# You might need to load X_mod here if it's not available in the current environment
# X_mod = pd.read_csv("/content/India Agriculture Crop Production(1).csv") # Adjust path if necessary
# You would also need to apply the same data cleaning steps as in the notebook
# to X_mod before extracting unique values.

# For demonstration purposes, let's use some sample data for dropdowns
# In a real application, load and clean the original data to get these values
try:
    # Assuming X_mod is available in the environment from the previous steps
    states = sorted(X_mod['State'].unique().tolist())
    districts = sorted(X_mod['District'].unique().tolist())
    crops = sorted(X_mod['Crop'].unique().tolist())
    seasons = sorted(X_mod['Season'].unique().tolist())
except NameError:
    st.error("Original data (X_mod) not found. Cannot populate dropdowns. Please ensure X_mod is loaded and processed.")
    st.stop()


state = st.selectbox("State", states)
district = st.selectbox("District", districts)
crop = st.selectbox("Crop", crops)
season = st.selectbox("Season", seasons)
area = st.number_input("Area (in Hectare)", min_value=0.0, format="%f")

# --- Prediction Button ---
if st.button("Predict"):
    # --- Prepare input data for the pipeline ---
    input_data = pd.DataFrame([[state, district, crop, season, area]],
                              columns=['State', 'District', 'Crop', 'Season', 'Area'])

    # --- Make Prediction using the pipeline ---
    # The pipeline handles preprocessing and prediction
    predicted_production = full_pipeline.predict(input_data)[0]

    # --- Calculate Yield ---
    predicted_yield = predicted_production / area if area > 0 else 0

    # --- Display Results ---
    st.header("Prediction Results")
    st.write(f"Predicted Production: {predicted_production:.2f}")
    st.write(f"Predicted Yield (Production/Area): {predicted_yield:.2f}")
