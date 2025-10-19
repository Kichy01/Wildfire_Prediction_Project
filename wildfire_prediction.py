# wildfire_prediction.py

# ==============================================================================
# 1. Introduction and Setup
# ==============================================================================
# SDG: 13 - Climate Action
# Problem: Predict the severity (burned area) of forest fires using
#          meteorological data to help allocate resources effectively.
# ML Approach: Supervised Learning (Regression) using a Random Forest Regressor.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

def main():
    """Main function to run the ML pipeline."""
    print("Starting the wildfire prediction pipeline...")

    # ==============================================================================
    # 2. Data Loading
    # ==============================================================================
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/forest-fires/forestfires.csv'
    try:
        df = pd.read_csv(url)
        print("Dataset loaded successfully from UCI repository.")
    except Exception as e:
        print(f"Could not load data from URL. Error: {e}")
        return

    # ==============================================================================
    # 3. Data Preprocessing
    # ==============================================================================
    # One-hot encode categorical features
    df = pd.get_dummies(df, columns=['month', 'day'], drop_first=True)

    # Log-transform the skewed 'area' target variable
    df['area_log'] = np.log1p(df['area'])

    # --- Visualize distribution (and save it) ---
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(df['area'], kde=True, bins=50)
    plt.title('Distribution of Burned Area (Original)')
    plt.xlabel('Area (hectares)')
    plt.subplot(1, 2, 2)
    sns.histplot(df['area_log'], kde=True, color='green', bins=50)
    plt.title('Distribution of Burned Area (Log-Transformed)')
    plt.xlabel('Log(1 + Area)')
    plt.tight_layout()
    plt.savefig('screenshots/data_distribution.png')
    print("Saved data distribution plot to 'screenshots/data_distribution.png'")
    plt.close()

    # Split data into features (X) and target (y)
    X = df.drop(['area', 'area_log'], axis=1)
    y = df['area_log']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Data split into {len(X_train)} training and {len(X_test)} testing samples.")

    # ==============================================================================
    # 4. Model Training
    # ==============================================================================
    print("Training the Random Forest Regressor model...")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    print("Model training complete.")

    # ==============================================================================
    # 5. Model Evaluation
    # ==============================================================================
    y_pred_log = rf_model.predict(X_test)

    # Reverse the log transformation to get results in original units
    y_test_original = np.expm1(y_test)
    y_pred_original = np.expm1(y_pred_log)

    # Calculate metrics
    r2 = r2_score(y_test, y_pred_log)
    mae = mean_absolute_error(y_test_original, y_pred_original)

    print("\n--- Model Evaluation ---")
    print(f"R-squared (R²): {r2:.3f}")
    print(f"Mean Absolute Error (MAE): {mae:.3f} hectares")
    print("------------------------")

    # --- Visualize results (and save it) ---
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=y_test_original, y=y_pred_original, alpha=0.7)
    plt.plot([0, max(y_test_original)], [0, max(y_test_original)], color='red', linestyle='--', lw=2)
    plt.xlabel("Actual Burned Area (hectares)")
    plt.ylabel("Predicted Burned Area (hectares)")
    plt.title("Actual vs. Predicted Burned Area")
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(1, max(y_test_original) + 10)
    plt.ylim(1, max(y_pred_original) + 10)
    plt.savefig('screenshots/results_plot.png')
    print("Saved results plot to 'screenshots/results_plot.png'")
    plt.close()

if __name__ == '__main__':
    # Create screenshots directory if it doesn't exist
    import os
    if not os.path.exists('screenshots'):
        os.makedirs('screenshots')
    main()