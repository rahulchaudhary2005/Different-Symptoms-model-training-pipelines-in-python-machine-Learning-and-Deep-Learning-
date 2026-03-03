# Kidney Disease Prediction Script (Fixed)

import numpy as np
import pandas as pd
import joblib


# Load saved files
model = joblib.load("kidney_model.pkl")
scaler = joblib.load("scaler.pkl")
num_imputer = joblib.load("num_imputer.pkl")
le = joblib.load("label_encoder.pkl")
features = joblib.load("feature_names.pkl")


# -------------------------------
# Example Patient Data
# -------------------------------

# ⚠️ ENTER ALL FEATURES IN SAME ORDER
# Replace values according to your dataset

patient_dict = {
    "age": 45,
    "bp": 80,
    "sg": 1.020,
    "al": 1,
    "su": 0,
    "bgr": 120,
    "bu": 40,
    "sc": 1.5,
    "hemo": 15.5,
    "pcv": 42,
    # Add ALL remaining columns here
}


# Convert to DataFrame
patient_df = pd.DataFrame([patient_dict])

# Reorder columns
patient_df = patient_df.reindex(columns=features)


# -------------------------------
# Handle Missing Values
# -------------------------------

patient_df[patient_df.columns] = num_imputer.transform(patient_df)


# -------------------------------
# Scale
# -------------------------------

patient_scaled = scaler.transform(patient_df)


# -------------------------------
# Predict
# -------------------------------

prediction = model.predict(patient_scaled)

result = le.inverse_transform(prediction)

print("\n--------------------")
print("Kidney Disease Prediction")
print("--------------------")

print("Result:", result[0])