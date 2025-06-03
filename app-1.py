import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta

# --- Configuration ---
ARTIFACTS_DIR = "app_artifacts"
MODEL_FILE = os.path.join(ARTIFACTS_DIR, "svr_model.pkl")
PREPROCESSOR_FILE = os.path.join(ARTIFACTS_DIR, "preprocessor.pkl")
FEATURE_COLS_FILE = os.path.join(ARTIFACTS_DIR, "feature_input_columns.pkl")
DATA_FILE = os.path.join(ARTIFACTS_DIR, "historical_data.csv") # This is the raw data file

# --- Global variables for app ---
TARGET_COL_APP = 'confirm_dengue' # Standardized name
METEO_COLS_APP = ['wsa', 'ta', 'ra', 'rha', 'psa', 'da'] # Standardized names
LAGS_TO_USE_APP = [1, 2, 3, 7, 14, 21, 30, 60]
ROLLING_WINDOWS_APP = [7, 14, 30, 60]


# --- Helper Functions (must align with model_builder.py) ---
def get_season_from_month_strict(month_val):
    """Determines season from month based on user's definition."""
    if month_val in [4, 5, 6]: return 'Summer'
    if month_val in [7, 8, 9, 10]: return 'Monsoon'
    if month_val in [11, 12, 1, 2]: return 'Winter'
    return 'UnknownSeason' # Should not be reached

def feature_engineer_app(df_input, target_col_name, meteo_cols_list):
    """Performs feature engineering for the app.
    Processes the combined dataframe (history + new input) to generate features,
    especially for the last row(s) corresponding to new input.
    """
    df_fe = df_input.copy()

    # Ensure date is index and sorted
    if not isinstance(df_fe.index, pd.DatetimeIndex):
        if 'date' in df_fe.columns:
            # Convert to datetime, coercing errors. App should validate date input.
            df_fe['date_col_temp'] = pd.to_datetime(df_fe['date'], errors='coerce')
            if df_fe['date_col_temp'].isnull().any():
                 st.error("Invalid date format encountered during feature engineering. Please ensure date is correct.")
                 return None # Indicate error
            df_fe.set_index('date_col_temp', inplace=True)
            if 'date' in df_fe.columns and df_fe.index.name == 'date_col_temp':
                 df_fe.drop(columns=['date'], inplace=True, errors='ignore')
        else:
            st.error("Date column missing for feature engineering in app.")
            return None
    df_fe.sort_index(inplace=True)

    # Lag features
    for col in [target_col_name] + meteo_cols_list:
        if col in df_fe.columns:
            for lag in LAGS_TO_USE_APP:
                df_fe[f'{col}_lag_{lag}'] = df_fe[col].shift(lag)

    # Rolling statistics
    for col in [target_col_name] + meteo_cols_list:
        if col in df_fe.columns:
            for window in ROLLING_WINDOWS_APP:
                df_fe[f'{col}_roll_mean_{window}'] = df_fe[col].shift(1).rolling(window=window, min_periods=1).mean()
                df_fe[f'{col}_roll_std_{window}'] = df_fe[col].shift(1).rolling(window=window, min_periods=1).std()

    # Calendar features
    df_fe['month'] = df_fe.index.month.astype(int)
    df_fe['year'] = df_fe.index.year.astype(int)
    df_fe['season'] = df_fe.index.month.map(get_season_from_month_strict)
    
    # Drop original calendar columns if they existed from raw input to avoid conflict
    original_calendar_cols_to_drop = ['YEAR', 'MONTH', 'DAY', 'Season']
    cols_present_in_df = df_fe.columns.tolist()
    for col_to_check in original_calendar_cols_to_drop:
        for actual_col in cols_present_in_df:
            if actual_col.lower() == col_to_check.lower():
                if actual_col.lower() != 'season': 
                    df_fe = df_fe.drop(columns=[actual_col], errors='ignore')
    return df_fe

# --- Load Artifacts ---
@st.cache_resource # Cache resource loading (model, preprocessor)
def load_artifacts_cached():
    if not all([os.path.exists(f) for f in [MODEL_FILE, PREPROCESSOR_FILE, FEATURE_COLS_FILE]]):
        st.error("Critical artifact files (model, preprocessor, or feature columns) are missing. "
                 "Please run the model_builder.py script first.")
        st.stop()
    model = joblib.load(MODEL_FILE)
    preprocessor = joblib.load(PREPROCESSOR_FILE)
    feature_input_cols = joblib.load(FEATURE_COLS_FILE)
    return model, preprocessor, feature_input_cols

@st.cache_data # Cache data loading (historical data)
def load_historical_data_cached(data_file_path):
    if not os.path.exists(data_file_path):
        st.error(f"Historical data file not found: {data_file_path}. Run model_builder.py.")
        # Create a minimal empty DataFrame with expected columns to prevent app from fully crashing
        # These column names should match what model_builder.py saves as historical_data.csv's header
        # For now, using a generic 'date' and target/meteo, but ideally, load actual header if possible
        # This is a fallback, the file *should* exist.
        fallback_cols = ['date', TARGET_COL_APP] + METEO_COLS_APP + ['year', 'month', 'day', 'season']
        return pd.DataFrame(columns=fallback_cols)

    # Read the CSV, ensuring date parsing and standardization of column names
    df_hist = pd.read_csv(data_file_path)
    df_hist.columns = df_hist.columns.str.replace(' ', '_').str.replace('-', '_').str.lower()
    if 'date' in df_hist.columns:
        df_hist['date'] = pd.to_datetime(df_hist['date'], errors='coerce')
    else:
        st.warning("No 'date' column in historical_data.csv after loading and cleaning names.")
    return df_hist

# --- Main App Logic ---
def run_app():
    st.set_page_config(layout="wide", page_title="Dengue Prediction App")
    st.title("Dengue Cases Prediction (7 Days Ahead)")

    model, preprocessor, feature_input_cols_for_model = load_artifacts_cached()
    df_historical_raw = load_historical_data_cached(DATA_FILE) # Raw data, original columns

    # Standardize column names of historical data for internal app use
    df_historical_app_use = df_historical_raw.copy()
    df_historical_app_use.columns = df_historical_app_use.columns.str.replace(' ', '_').str.replace('-', '_').str.lower()
    if 'date' in df_historical_app_use.columns:
         df_historical_app_use['date'] = pd.to_datetime(df_historical_app_use['date'], errors='coerce')


    # --- Sidebar for Inputs ---
    st.sidebar.header("Input Today's Data")
    
    default_input_date = datetime.today()
    if 'date' in df_historical_app_use.columns and not df_historical_app_use['date'].dropna().empty:
        last_known_date = df_historical_app_use['date'].dropna().max()
        default_input_date = last_known_date + timedelta(days=1)

    input_date_dt = st.sidebar.date_input("Date", value=default_input_date)
    input_date_pd = pd.to_datetime(input_date_dt) # Convert to pandas Timestamp

    input_confirm_dengue = st.sidebar.number_input(
        f"Today's Confirmed Dengue ({TARGET_COL_APP})", 
        min_value=0, value=10, step=1
    )

    st.sidebar.subheader("Meteorological Variables:")
    input_meteo_vars_dict = {}
    # Default values for meteo vars, can be adjusted
    default_meteo_vals = {'wsa': 2.0, 'ta': 28.0, 'da': 22.0, 'rha': 80.0, 'ra': 5.0, 'psa': 101.0}
    for var_name in METEO_COLS_APP:
        input_meteo_vars_dict[var_name] = st.sidebar.number_input(
            f"{var_name.upper()}", 
            value=default_meteo_vals.get(var_name, 0.0), 
            format="%.2f"
        )

    if st.sidebar.button("Predict 7 Days Ahead"):
        # 1. Prepare new input row (using standardized column names)
        new_input_data_dict = {'date': input_date_pd, TARGET_COL_APP: float(input_confirm_dengue)}
        new_input_data_dict.update(input_meteo_vars_dict)
        
        new_input_df_single_row = pd.DataFrame([new_input_data_dict])
        new_input_df_single_row['date'] = pd.to_datetime(new_input_df_single_row['date'])

        # 2. Combine with historical data for feature engineering
        # Use only essential base columns for feature engineering (date, target, meteo)
        # This ensures consistency if historical_data.csv has other cols.
        base_cols_for_fe = ['date', TARGET_COL_APP] + METEO_COLS_APP
        
        # Ensure df_historical_app_use has these base columns
        df_hist_for_concat = df_historical_app_use.copy()
        for col in base_cols_for_fe: # Ensure all base columns are present, add if missing (as NaN)
            if col not in df_hist_for_concat.columns:
                df_hist_for_concat[col] = np.nan
        df_hist_for_concat = df_hist_for_concat[base_cols_for_fe]


        combined_df_for_fe = pd.concat([df_hist_for_concat, new_input_df_single_row[base_cols_for_fe]], ignore_index=True)
        combined_df_for_fe['date'] = pd.to_datetime(combined_df_for_fe['date'])
        # Remove duplicates, keeping the latest entry for a given date (the new input)
        combined_df_for_fe = combined_df_for_fe.drop_duplicates(subset=['date'], keep='last')
        # Set date as index for feature_engineer_app
        combined_df_for_fe.set_index('date', inplace=True)
        combined_df_for_fe.sort_index(inplace=True)
        
        # 3. Feature Engineer on combined data
        df_featured_full = feature_engineer_app(combined_df_for_fe, TARGET_COL_APP, METEO_COLS_APP)

        if df_featured_full is None: # Error in feature engineering
            st.error("Prediction aborted due to feature engineering error.")
            st.stop()

        # 4. Extract the feature row for the new input date
        try:
            # input_date_pd is the specific date we want features for
            input_features_for_pred_row = df_featured_full.loc[[input_date_pd]]
        except KeyError:
            st.error(f"Features for date {input_date_pd.strftime('%Y-%m-%d')} could not be extracted after engineering. "
                     "This might be due to this date being too early for lag/rolling features, or an issue with the date input.")
            st.dataframe(df_featured_full.tail()) # Show tail for debugging
            st.stop()
        
        # Align with the columns expected by the preprocessor/model
        X_predict_raw = input_features_for_pred_row[feature_input_cols_for_model]

        if X_predict_raw.isnull().any().any():
            st.warning(f"The feature row for {input_date_pd.strftime('%Y-%m-%d')} contains NaNs after engineering. "
                       "Prediction might be unreliable or fail. This can happen with insufficient historical data for lags/rolling windows.")
            st.dataframe(X_predict_raw[X_predict_raw.isnull().any(axis=1)])
            # Decide whether to stop or proceed with NaNs (preprocessor might handle some)
            # For now, let's proceed but with caution.

        # 5. Preprocess the feature row
        X_predict_processed = preprocessor.transform(X_predict_raw)

        # 6. Predict
        prediction = model.predict(X_predict_processed)
        predicted_cases = max(0, int(round(prediction[0])))

        prediction_for_date = input_date_pd + timedelta(days=7)
        st.subheader(f"Prediction for {prediction_for_date.strftime('%Y-%m-%d')}:")
        st.metric(label="Predicted Dengue Cases", value=f"{predicted_cases} cases")

        # 7. Data and Model Update Logic
        last_hist_date_in_file = None
        if 'date' in df_historical_raw.columns and not df_historical_raw['date'].dropna().empty:
             last_hist_date_in_file = pd.to_datetime(df_historical_raw['date'].dropna()).max()
        
        is_new_date = last_hist_date_in_file is None or input_date_pd > last_hist_date_in_file

        if is_new_date:
            st.info(f"New data for {input_date_pd.strftime('%Y-%m-%d')} provided.")
            
            # Prepare the new row for storage in historical_data.csv (matching its original format)
            new_row_to_store = {'date': input_date_pd.strftime('%Y-%m-%d')} # Format as string for CSV
            new_row_to_store[TARGET_COL_APP] = input_confirm_dengue # Already standardized name
            for var, val in input_meteo_vars_dict.items(): # Already standardized names
                new_row_to_store[var] = val
            
            # Add original calendar fields if they were in the initial de.csv (model_builder saves them)
            # These are 'YEAR', 'MONTH', 'DAY', 'Season' (potentially mixed case in original de.csv)
            # We need to match the column names of the existing DATA_FILE.
            # For simplicity, assume model_builder.py saved historical_data.csv with lowercased column names
            # that include year, month, day, season if they were in original de.csv.
            # If not, the feature_engineer in model_builder would have derived them.
            # The safest is to add what we know:
            new_row_to_store['year'] = input_date_pd.year
            new_row_to_store['month'] = input_date_pd.month
            new_row_to_store['day'] = input_date_pd.day
            new_row_to_store['season'] = get_season_from_month_strict(input_date_pd.month)
            
            # Create a DataFrame for the new entry, ensuring columns match historical_data.csv
            # Load the header of the existing CSV to get exact column names
            try:
                existing_csv_header = pd.read_csv(DATA_FILE, nrows=0).columns.tolist()
            except FileNotFoundError: # Should not happen if model_builder ran
                existing_csv_header = [col.lower().replace(' ','_') for col in new_row_to_store.keys()] # Fallback

            df_new_entry_for_csv = pd.DataFrame(columns=existing_csv_header)
            
            # Map our standardized data to the CSV's column names (handle case differences)
            temp_new_row_data_standardized = {k.lower().replace(' ','_'): v for k,v in new_row_to_store.items()}

            for col_in_csv in existing_csv_header:
                standardized_col_in_csv = col_in_csv.lower().replace(' ','_')
                if standardized_col_in_csv in temp_new_row_data_standardized:
                    df_new_entry_for_csv.loc[0, col_in_csv] = temp_new_row_data_standardized[standardized_col_in_csv]
                else: # If CSV has columns not in our new data (e.g. old 'Season' vs derived 'season')
                    df_new_entry_for_csv.loc[0, col_in_csv] = np.nan


            df_new_entry_for_csv.to_csv(DATA_FILE, mode='a', header=False, index=False)
            st.success(f"Data for {input_date_pd.strftime('%Y-%m-%d')} added to historical records.")
            st.cache_data.clear() # Clear cache for load_historical_data_cached
            # df_historical_raw = load_historical_data_cached(DATA_FILE) # Reload for display

            # Retraining logic: "once a month"
            needs_retrain = False
            if last_hist_date_in_file is None: # First data point after initial model
                needs_retrain = True
            elif input_date_pd.year > last_hist_date_in_file.year or \
                 (input_date_pd.year == last_hist_date_in_file.year and input_date_pd.month > last_hist_date_in_file.month):
                needs_retrain = True

            if needs_retrain:
                st.info("New month's data boundary crossed. Retraining model with all available data...")
                try:
                    # Reload all historical data (now including the new entry)
                    df_full_updated_raw = pd.read_csv(DATA_FILE)
                    df_full_updated_app_use = df_full_updated_raw.copy()
                    df_full_updated_app_use.columns = df_full_updated_app_use.columns.str.replace(' ', '_').str.replace('-', '_').str.lower()
                    if 'date' in df_full_updated_app_use.columns:
                        df_full_updated_app_use['date'] = pd.to_datetime(df_full_updated_app_use['date'])
                        # Set index for feature engineering
                        df_full_updated_app_use.set_index('date', inplace=True, drop=False) # Keep date column for now
                        df_full_updated_app_use.sort_index(inplace=True)


                    # Feature engineer on the entire updated dataset
                    df_fe_retrain = feature_engineer_app(df_full_updated_app_use, TARGET_COL_APP, METEO_COLS_APP)
                    if df_fe_retrain is None:
                        st.error("Retraining failed during feature engineering.")
                        st.stop()
                    
                    df_fe_retrain['target_lead_7'] = df_fe_retrain[TARGET_COL_APP].shift(-7)
                    df_retrain_final = df_fe_retrain.dropna().copy()

                    if df_retrain_final.empty or len(df_retrain_final) < 2: # Need at least some data
                        st.warning("Not enough data to retrain model after processing.")
                    else:
                        y_retrain = df_retrain_final['target_lead_7']
                        # Ensure X_retrain uses the same columns as feature_input_cols_for_model
                        X_retrain = df_retrain_final[feature_input_cols_for_model] 
                        
                        # Re-fit preprocessor on ALL new data
                        X_retrain_processed = preprocessor.fit_transform(X_retrain)
                        
                        # Re-train model
                        model.fit(X_retrain_processed, y_retrain)

                        joblib.dump(model, MODEL_FILE)
                        joblib.dump(preprocessor, PREPROCESSOR_FILE) # Save re-fitted preprocessor
                        
                        st.success("Model retrained and artifacts updated successfully!")
                        st.cache_resource.clear() # Clear cache for load_artifacts_cached
                        # model, preprocessor, _ = load_artifacts_cached() # Reload for current session
                except Exception as e:
                    st.error(f"Error during model retraining: {e}")
                    st.exception(e) # Show full traceback for debugging
            else:
                st.info("Data updated. Model retraining not scheduled for this update.")

        elif input_date_pd <= last_hist_date_in_file:
            st.warning(f"Data for {input_date_pd.strftime('%Y-%m-%d')} already exists or is older than the latest record. "
                       "Prediction shown, but data/model not updated.")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("This app predicts dengue cases 7 days in advance.")

    # Display some info about the loaded data
    st.subheader("Historical Data Overview")
    current_data_for_display = load_historical_data_cached(DATA_FILE) # Get latest
    if not current_data_for_display.empty and 'date' in current_data_for_display.columns and not current_data_for_display['date'].dropna().empty:
        min_date_display = pd.to_datetime(current_data_for_display['date'].dropna()).min().strftime('%Y-%m-%d')
        max_date_display = pd.to_datetime(current_data_for_display['date'].dropna()).max().strftime('%Y-%m-%d')
        st.write(f"Historical data from {min_date_display} to {max_date_display}.")
        st.write(f"Total records: {len(current_data_for_display)}")
        if st.checkbox("Show raw historical data sample (last 5 rows)"):
            st.dataframe(current_data_for_display.tail())
    else:
        st.write("No historical data loaded or 'date' column is problematic.")

if __name__ == "__main__":
    run_app()
