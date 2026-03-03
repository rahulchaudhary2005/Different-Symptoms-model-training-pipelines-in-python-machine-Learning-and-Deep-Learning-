# Kidney Disease Prediction System with Risk Assessment (Final Safe Version)

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# ------------------------------------------------
# Risk Level Function
# ------------------------------------------------
def risk_level(creatinine, urea, hemo):
    if creatinine > 5 or urea > 150 or hemo < 8:
        return "HIGH RISK"
    elif creatinine > 2:
        return "MODERATE RISK"
    else:
        return "LOW RISK"


# ------------------------------------------------
# STEP 1: Load Dataset
# ------------------------------------------------
df = pd.read_csv("kidney_disease_cleaned.csv")

print("Dataset Loaded Successfully!")
print("Shape:", df.shape)


# ------------------------------------------------
# STEP 2: Basic Cleaning
# ------------------------------------------------

df.replace("?", np.nan, inplace=True)
df.columns = df.columns.str.strip()

# Convert numeric safely
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")


# ------------------------------------------------
# STEP 3: Separate Features & Target
# ------------------------------------------------

TARGET_COL = "class"   # Change if needed

X = df.drop(TARGET_COL, axis=1)
y = df[TARGET_COL]


# ------------------------------------------------
# STEP 4: Remove Fully Empty Columns
# ------------------------------------------------

empty_cols = X.columns[X.isnull().all()]

if len(empty_cols) > 0:
    print("Removing Empty Columns:", list(empty_cols))
    X.drop(columns=empty_cols, inplace=True)


# ------------------------------------------------
# STEP 5: Identify Column Types
# ------------------------------------------------

num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

print("Numeric Columns:", len(num_cols))
print("Categorical Columns:", len(cat_cols))


# ------------------------------------------------
# STEP 6: Handle Missing Values (SAFE)
# ------------------------------------------------

num_imputer = None
cat_imputer = None

# Numeric → Median
if len(num_cols) > 0:
    num_imputer = SimpleImputer(strategy="median")
    X[num_cols] = num_imputer.fit_transform(X[num_cols])

# Categorical → Mode
if len(cat_cols) > 0:
    cat_imputer = SimpleImputer(strategy="most_frequent")
    X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])


# ------------------------------------------------
# STEP 7: Encode Categorical Features (SAFE)
# ------------------------------------------------

le = LabelEncoder()

if len(cat_cols) > 0:
    for col in cat_cols:
        X[col] = le.fit_transform(X[col])

# Encode target
y = le.fit_transform(y)


# ------------------------------------------------
# STEP 8: Feature Scaling
# ------------------------------------------------

scaler = StandardScaler()
X = scaler.fit_transform(X)


# ------------------------------------------------
# STEP 9: Train-Test Split
# ------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    
)


# ------------------------------------------------
# STEP 10: Train Model
# ------------------------------------------------

model = RandomForestClassifier(
    n_estimators=400,
    max_depth=15,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Model Training Completed!")


# ------------------------------------------------
# STEP 11: Evaluate Model
# ------------------------------------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n--------------------")
print("Model Performance")
print("--------------------")

print("Accuracy:", round(accuracy * 100, 2), "%\n")
print("Classification Report:\n")
print(classification_report(y_test, y_pred))


# ------------------------------------------------
# STEP 12: Save Everything
# ------------------------------------------------

# Save feature order
joblib.dump(list(X.columns), "feature_names.pkl")

joblib.dump(model, "kidney_model.pkl")
joblib.dump(scaler, "scaler.pkl")


if num_imputer is not None:
    joblib.dump(num_imputer, "num_imputer.pkl")

if cat_imputer is not None:
    joblib.dump(cat_imputer, "cat_imputer.pkl")

joblib.dump(le, "label_encoder.pkl")

print("\nModel & Preprocessors Saved Successfully!")


# ------------------------------------------------
# STEP 13: Sample Risk Test
# ------------------------------------------------

serum_creatinine = 3.2
blood_urea = 120
hemoglobin = 10.5

risk = risk_level(serum_creatinine, blood_urea, hemoglobin)

print("\n--------------------")
print("Patient Risk Level")
print("--------------------")
print("Risk Status:", risk)