import pandas as pd
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ─── Helpers ─────────────────────────────────────────────────
def make_sample_df():
    return pd.DataFrame({
        "customerID":      ["1111-AAAAA", "2222-BBBBB", "3333-CCCCC"],
        "gender":          ["Male", "Female", "Male"],
        "SeniorCitizen":   [0, 1, 0],
        "Partner":         ["Yes", "No", "Yes"],
        "Dependents":      ["No", "No", "Yes"],
        "tenure":          [12, 24, 1],
        "PhoneService":    ["Yes", "Yes", "No"],
        "MultipleLines":   ["No", "Yes", "No phone service"],
        "InternetService": ["DSL", "Fiber optic", "No"],
        "OnlineSecurity":  ["Yes", "No", "No internet service"],
        "OnlineBackup":    ["No", "Yes", "No internet service"],
        "DeviceProtection":["No", "No", "No internet service"],
        "TechSupport":     ["Yes", "No", "No internet service"],
        "StreamingTV":     ["No", "Yes", "No internet service"],
        "StreamingMovies": ["No", "No", "No internet service"],
        "Contract":        ["Month-to-month", "One year", "Two year"],
        "PaperlessBilling":["Yes", "No", "Yes"],
        "PaymentMethod":   ["Electronic check", "Mailed check", "Bank transfer (automatic)"],
        "MonthlyCharges":  [65.5, 89.0, 20.0],
        "TotalCharges":    ["786.0", "2136.0", " "],
        "Churn":           ["Yes", "No", "No"],
    })

# ─── Tests ───────────────────────────────────────────────────
def test_rows_preserved():
    from src.preprocess import clean_data
    df = make_sample_df()
    result = clean_data(df)
    assert len(result) == 3

def test_customerid_dropped():
    from src.preprocess import clean_data
    df = make_sample_df()
    result = clean_data(df)
    assert "customerID" not in result.columns

def test_churn_is_binary():
    from src.preprocess import clean_data
    df = make_sample_df()
    result = clean_data(df)
    assert set(result["Churn"].unique()).issubset({0, 1})

def test_total_charges_no_nulls():
    from src.preprocess import clean_data
    df = make_sample_df()
    result = clean_data(df)
    assert result["TotalCharges"].isnull().sum() == 0

def test_total_charges_is_numeric():
    from src.preprocess import clean_data
    df = make_sample_df()
    result = clean_data(df)
    assert pd.api.types.is_numeric_dtype(result["TotalCharges"])

def test_no_object_columns_remain():
    from src.preprocess import clean_data
    df = make_sample_df()
    result = clean_data(df)
    object_cols = result.select_dtypes(include="object").columns.tolist()
    assert len(object_cols) == 0, f"Object columns still present: {object_cols}"