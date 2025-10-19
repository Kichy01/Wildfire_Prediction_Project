# app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# ==============================================================================
# Caching the Model
# ==============================================================================
# The @st.cache_resource decorator tells Streamlit to run this function only once
# and store the result in memory. This is perfect for loading models or data,
# so you don't have to retrain your model every time the user changes a slider.

@st.cache_resource
def train_model():
    """
    This function loads the data, preprocesses it, and trains the
    Random Forest model. It returns the trained model and the list of
    feature columns the model was trained on.
    """
    # Load the dataset from the URL
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/forest-fires/forestfires.csv'
    df = pd.read_csv(url)

    # --- Preprocessing ---
    # One-hot encode categorical features
    df = pd.get_dummies(df, columns=['month', 'day'], drop_first=True)
    # Log-transform the skewed 'area' target variable
    df['area_log'] = np.log1p(df['area'])

    # Define features (X) and target (y)
    features = df.drop(['area', 'area_log'], axis=1)
    target = df['area_log']

    # Get the column names for later use
    feature_columns = features.columns

    # --- Model Training ---
    # We train on the FULL dataset here to make the app's predictions as robust as possible.
    # In a real-world scenario, you'd save a model trained on a full, clean dataset.
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(features, target)

    return model, feature_columns

# ==============================================================================
# Main App Interface
# ==============================================================================

# --- Page Configuration ---
st.set_page_config(
    page_title="Wildfire Severity Prediction",
    page_icon="🔥",
    layout="centered"
)

# --- Load Model ---
# This will run the function above on the first run and then use the cached version.
model, feature_columns = train_model()

# --- App Title and Description ---
st.title("🔥 AI for Climate Action: Wildfire Severity Prediction")
st.markdown("""
This app addresses **UN SDG 13: Climate Action** by predicting the burned area of forest fires.
Use the sliders and dropdowns on the left to input conditions for a fire and see the model's prediction for its potential severity.
""")

# --- Sidebar for User Inputs ---
st.sidebar.header("Input Fire Conditions")

# Create all the input widgets in the sidebar
# Note: min, max, and default values are chosen based on the dataset's statistics
X_coord = st.sidebar.number_input('Geographical X Coordinate', min_value=1, max_value=9, value=4)
Y_coord = st.sidebar.number_input('Geographical Y Coordinate', min_value=2, max_value=9, value=5)

# Dropdowns for month and day
# Note: These must match the original dataset categories
month_options = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
day_options = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']
month = st.sidebar.selectbox('Month', options=month_options, index=7) # Default to 'aug'
day = st.sidebar.selectbox('Day', options=day_options, index=4) # Default to 'fri'

# Sliders for Fire Weather Index (FWI) components and meteorological data
ffmc = st.sidebar.slider('FFMC Index', 80.0, 100.0, 91.5)
dmc = st.sidebar.slider('DMC Index', 1.0, 300.0, 142.4)
dc = st.sidebar.slider('DC Index', 7.0, 900.0, 681.8)
isi = st.sidebar.slider('ISI Index', 0.0, 60.0, 9.2)
temp = st.sidebar.slider('Temperature (°C)', 2.0, 34.0, 19.3)
rh = st.sidebar.slider('Relative Humidity (%)', 15, 100, 42)
wind = st.sidebar.slider('Wind Speed (km/h)', 0.0, 10.0, 4.0)
rain = st.sidebar.slider('Rain (mm/m2)', 0.0, 7.0, 0.0)

# --- Prediction Logic ---
if st.sidebar.button('**Predict Fire Severity**'):
    # 1. Create a dictionary with user inputs
    user_input = {
        'X': X_coord, 'Y': Y_coord, 'FFMC': ffmc, 'DMC': dmc,
        'DC': dc, 'ISI': isi, 'temp': temp, 'RH': rh,
        'wind': wind, 'rain': rain
    }

    # 2. Create an empty DataFrame with the same columns as the training data, filled with zeros
    input_df = pd.DataFrame(columns=feature_columns)
    input_df.loc[0] = 0

    # 3. Fill in the numerical values from user input
    for key, value in user_input.items():
        input_df[key] = value

    # 4. Handle the one-hot encoded categorical columns for month and day
    # We find the column that matches the user's choice and set its value to 1
    month_col = f'month_{month}'
    if month_col in input_df.columns:
        input_df[month_col] = 1

    day_col = f'day_{day}'
    if day_col in input_df.columns:
        input_df[day_col] = 1

    # Ensure the column order is exactly the same as during training
    input_df = input_df[feature_columns]

    # 5. Make the prediction
    prediction_log = model.predict(input_df)
    
    # 6. Reverse the log transformation to get the result in hectares
    prediction_hectares = np.expm1(prediction_log[0])

    # --- Display the Prediction ---
    st.subheader("📈 Prediction Result")
    st.metric(
        label="Predicted Burned Area",
        value=f"{prediction_hectares:.2f} hectares"
    )

    # Provide a simple interpretation
    if prediction_hectares < 1:
        st.info("The model predicts a **low severity** fire (less than 1 hectare).")
    elif prediction_hectares < 50:
        st.warning("The model predicts a **moderate severity** fire.")
    else:
        st.error("The model predicts a **high severity** fire. Urgent action may be required.")