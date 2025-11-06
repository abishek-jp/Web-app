import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the trained model and preprocessor
try:
    model = joblib.load('model.pkl')
    ohe = joblib.load('onehotencoder.joblib')
    categorical_columns = joblib.load('categorical_columns.joblib')
except FileNotFoundError:
    st.error("Model or preprocessor files not found. Make sure 'model.pkl', 'onehotencoder.joblib', and 'categorical_columns.joblib' are in the same directory.")
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
    # --- Preprocessing Input ---
    input_data = pd.DataFrame([[state, district, crop, season, area]],
                              columns=['State', 'District', 'Crop', 'Season', 'Area'])

    # Separate categorical and numerical columns
    input_categorical = input_data[categorical_columns]
    input_numerical = input_data.drop(categorical_columns, axis=1)

    # Apply OneHotEncoder
    # Handle potential errors if a category is not seen during training
    try:
        input_categorical_encoded = ohe.transform(input_categorical)
        input_categorical_encoded_df = pd.DataFrame(input_categorical_encoded, columns=ohe.get_feature_names_out(categorical_columns))
    except ValueError as e:
        st.error(f"Error during encoding: {e}. This might be due to a new category in the input data that was not present during training.")
        st.stop()


    # Concatenate numerical and encoded categorical features
    # Ensure columns match the training data order if necessary (though OHE usually handles this)
    # A more robust approach would be to save the training column order
    # For simplicity here, we assume the order from get_feature_names_out is consistent
    input_processed = pd.concat([input_numerical.reset_index(drop=True), input_categorical_encoded_df.reset_index(drop=True)], axis=1)

    # Ensure all columns from training are present, add missing columns with 0
    # This is crucial if handle_unknown='ignore' is used and new categories appear
    training_columns = model.get_booster().feature_names # Get feature names from the trained model
    missing_cols = set(training_columns) - set(input_processed.columns)
    for c in missing_cols:
        input_processed[c] = 0
    # Ensure the order of columns is the same as the training data
    input_processed = input_processed[training_columns]


    # --- Make Prediction ---
    predicted_production = model.predict(input_processed)[0]

    # --- Calculate Yield ---
    predicted_yield = predicted_production / area if area > 0 else 0

    # --- Display Results ---
    st.header("Prediction Results")
    st.write(f"Predicted Production: {predicted_production:.2f}")
    st.write(f"Predicted Yield (Production/Area): {predicted_yield:.2f}")