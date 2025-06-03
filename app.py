import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta
from pathlib import Path # Import pathlib

# --- Configuration ---
# ARTIFACTS_DIR is now defined dynamically relative to the script file

# --- Load Artifacts ---
@st.cache_resource 
def load_artifacts():
    """Loads the pre-trained model, preprocessor, and other necessary artifacts."""
    try:
        script_dir = Path(__file__).resolve().parent
    except NameError:
        script_dir = Path.cwd() 

    artifacts_base_path = script_dir / 'model_artifacts' 
    model_filename_to_load = 'best_voting_model.pkl' 

    paths_to_check = {
        "model": artifacts_base_path / model_filename_to_load,
        "preprocessor": artifacts_base_path / 'preprocessor.pkl',
        "features_before": artifacts_base_path / 'feature_names_before_preprocessing.pkl',
        "train_means": artifacts_base_path / 'train_numerical_feature_means.pkl',
        "meteo_cols": artifacts_base_path / 'meteo_cols_used.pkl',
        "target_col": artifacts_base_path / 'target_col_name.pkl',
        "test_mae": artifacts_base_path / 'test_mae_voting.pkl' # New artifact for prediction range
    }

    missing_files_details = []
    for name, path_obj in paths_to_check.items():
        if not path_obj.exists():
            relative_path_for_error = Path('model_artifacts') / path_obj.name
            missing_files_details.append(f"{name} (expected at ./{relative_path_for_error})")

    if missing_files_details:
        st.error(f"Error: One or more artifact files are missing. Please ensure these files exist in your repository's 'model_artifacts' folder (relative to your main app script) and are committed to GitHub:")
        for detail in missing_files_details:
            st.error(f"- {detail}")
        return None
    
    try:
        loaded_artifacts = {}
        loaded_artifacts["model"] = joblib.load(paths_to_check["model"].open('rb'))
        loaded_artifacts["preprocessor"] = joblib.load(paths_to_check["preprocessor"].open('rb'))
        loaded_artifacts["feature_names_before"] = joblib.load(paths_to_check["features_before"].open('rb'))
        loaded_artifacts["train_numerical_means"] = joblib.load(paths_to_check["train_means"].open('rb'))
        loaded_artifacts["meteo_cols"] = joblib.load(paths_to_check["meteo_cols"].open('rb'))
        loaded_artifacts["target_col"] = joblib.load(paths_to_check["target_col"].open('rb'))
        loaded_artifacts["test_mae"] = joblib.load(paths_to_check["test_mae"].open('rb')) # Load the MAE
        
        return loaded_artifacts
    except Exception as e:
        st.error(f"Error loading artifacts after confirming existence: {e}")
        st.error(f"An error occurred while trying to open/read the .pkl files. Ensure they are not corrupted and were generated correctly.")
        return None

# --- Helper function to get season ---
def get_season_for_input(month_num):
    if month_num in [12, 1, 2]: return 'Winter'
    if month_num in [3, 4, 5]: return 'Spring'
    if month_num in [6, 7, 8]: return 'Summer'
    return 'Autumn'

# --- Streamlit App UI and Logic ---
st.set_page_config(layout="wide")
st.title("Dengue Case Prediction (7 Days Ahead)")
st.markdown("""
This app predicts the number of dengue cases 7 days from a specified 'current date'.
It uses a pre-trained Voting Regressor model. Provide current data and data from 7 days ago.
""")

artifacts = load_artifacts()

if artifacts:
    st.sidebar.success("Model and artifacts loaded successfully!")

    model = artifacts["model"]
    preprocessor = artifacts["preprocessor"]
    feature_names_before_preprocessing = artifacts["feature_names_before"]
    train_numerical_feature_means = artifacts["train_numerical_means"]
    METEO_COLS_USED = artifacts["meteo_cols"]
    TARGET_COL_NAME = artifacts["target_col"] 
    TEST_MAE_VOTING = artifacts["test_mae"] # Get the loaded MAE

    st.sidebar.header("Input Data for Prediction")
    current_input_date = st.sidebar.date_input("Current Date for Input Data", datetime.today())
    
    st.sidebar.subheader("Today's Data (Day D)")
    current_dengue_cases_default = int(train_numerical_feature_means.get(TARGET_COL_NAME, 50))
    current_dengue_cases = st.sidebar.number_input(f"{TARGET_COL_NAME.replace('_', ' ').title()} (Today)", min_value=0, value=current_dengue_cases_default)
    
    current_meteo_inputs = {}
    for col in METEO_COLS_USED:
        default_val = train_numerical_feature_means.get(col, 20.0)
        current_meteo_inputs[col] = st.sidebar.number_input(f"{col.upper()} (Today)", value=float(default_val), format="%.2f")

    st.sidebar.subheader("Data from 7 Days Ago (Day D-7)")
    lag7_dengue_cases_default = int(train_numerical_feature_means.get(f"{TARGET_COL_NAME}_lag_7", 40))
    lag7_dengue_cases = st.sidebar.number_input(f"{TARGET_COL_NAME.replace('_', ' ').title()} (7 Days Ago)", min_value=0, value=lag7_dengue_cases_default)
    
    lag7_meteo_inputs = {}
    for col in METEO_COLS_USED:
        default_val = train_numerical_feature_means.get(f"{col}_lag_7", 18.0)
        lag7_meteo_inputs[f"{col}_lag_7"] = st.sidebar.number_input(f"{col.upper()} (7 Days Ago)", value=float(default_val), format="%.2f")

    if st.sidebar.button("Predict Dengue Cases for 7 Days Ahead"):
        input_data_dict = {}

        for f_name in feature_names_before_preprocessing:
            if f_name in train_numerical_feature_means:
                input_data_dict[f_name] = train_numerical_feature_means[f_name]
            else:
                input_data_dict[f_name] = np.nan 

        input_data_dict['month'] = current_input_date.month
        input_data_dict['year'] = current_input_date.year
        input_data_dict['day_of_week'] = current_input_date.strftime('%A')
        input_data_dict['season'] = get_season_for_input(current_input_date.month)

        input_data_dict[TARGET_COL_NAME] = float(current_dengue_cases)
        for col_meteo, val_meteo in current_meteo_inputs.items():
            if col_meteo in input_data_dict:
                 input_data_dict[col_meteo] = float(val_meteo)

        input_data_dict[f'{TARGET_COL_NAME}_lag_7'] = float(lag7_dengue_cases)
        for col_meteo_lag7, val_meteo_lag7 in lag7_meteo_inputs.items():
             if col_meteo_lag7 in input_data_dict: 
                input_data_dict[col_meteo_lag7] = float(val_meteo_lag7)
        
        input_df = pd.DataFrame([input_data_dict], columns=feature_names_before_preprocessing)
        
        if input_df.isnull().any().any():
            for col_idx, col_name in enumerate(input_df.columns): 
                if input_df[col_name].isnull().any(): 
                    if col_name in train_numerical_feature_means:
                        input_df[col_name] = input_df[col_name].fillna(train_numerical_feature_means[col_name])
                    else: 
                        st.error(f"Feature '{col_name}' is still NaN after initial imputation and has no mean value. Prediction might be inaccurate. Filling with 0 as last resort.")
                        input_df[col_name] = input_df[col_name].fillna(0)


        st.subheader("Prediction Results")
        
        num_imputed_features = sum(1 for f_name in feature_names_before_preprocessing 
                                   if f_name not in current_meteo_inputs and 
                                      f_name not in lag7_meteo_inputs and    
                                      f_name != TARGET_COL_NAME and               
                                      f_name != f'{TARGET_COL_NAME}_lag_7' and    
                                      f_name not in ['month', 'year', 'day_of_week', 'season'] and 
                                      f_name in train_numerical_feature_means) 

        if num_imputed_features > 0:
            st.info(f"""
                **Note on Input Simplification:** For this prediction, {num_imputed_features} features 
                (e.g., lags like `_lag_14`, `_lag_30`, and all rolling statistics like `_roll_mean_7`) 
                were not directly requested in the UI. These have been automatically filled using their 
                average values from the model's training data.
            """)
        
        try:
            input_processed = preprocessor.transform(input_df)
            prediction = model.predict(input_processed)
            predicted_cases = max(0, int(round(prediction[0]))) 

            # Calculate prediction range using Test MAE
            lower_bound = max(0, int(round(predicted_cases - TEST_MAE_VOTING)))
            upper_bound = int(round(predicted_cases + TEST_MAE_VOTING))

            prediction_date = current_input_date + timedelta(days=7)
            
            st.metric(
                label=f"Predicted {TARGET_COL_NAME.replace('_', ' ').title()} for {prediction_date.strftime('%Y-%m-%d')}",
                value=predicted_cases
            )
            st.markdown(f"**Expected Range (based on Test MAE ± {TEST_MAE_VOTING:.2f}):** `{lower_bound} - {upper_bound}` cases")
            
            st.markdown("---")
            st.markdown("#### Snapshot of Key Features Used for Prediction (after any imputation):")
            
            display_cols = []
            display_cols.extend([TARGET_COL_NAME] + METEO_COLS_USED) 
            display_cols.extend([f"{TARGET_COL_NAME}_lag_7"] + [f"{m}_lag_7" for m in METEO_COLS_USED]) 
            display_cols.extend(['month', 'year', 'day_of_week', 'season']) 
            if METEO_COLS_USED: 
                example_roll_col = f"{METEO_COLS_USED[0]}_roll_mean_7" 
                if example_roll_col in input_df.columns:
                    display_cols.append(example_roll_col)

            display_cols_unique = sorted(list(set(d for d in display_cols if d in input_df.columns)))
            if display_cols_unique:
                st.dataframe(input_df[display_cols_unique].style.format(precision=2))
            else:
                st.write("No key features available to display based on current configuration.")


        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.error("Details of the input DataFrame being processed (ensure all columns are correct):")
            st.dataframe(input_df)


else:
    st.error("App initialization failed: Could not load model artifacts. Please check the messages above and ensure your 'model_artifacts' folder and its contents are correctly placed in your GitHub repository relative to this script.")

st.markdown("---")
st.markdown("App using pre-trained model.")
