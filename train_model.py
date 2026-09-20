"""
train_model.py
---------------
Reproduces the pipeline from the "Linear, Ridge & Lasso Regression" notebook
and saves everything the Streamlit app needs (models, label encoders, feature
order, numeric ranges and evaluation metrics) into one file: car_price_bundle.pkl

Run this ONCE before deploying the app:
    python train_model.py

It will try to pull the dataset from Kaggle via kagglehub. If that isn't
available (no Kaggle credentials / no internet on the deploy machine), it
will fall back to a local CSV named "1.04. Real-life example.csv" placed in
this same folder (download it manually from the Kaggle dataset page:
smritisingh1997/car-salescsv).
"""

import os
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

FILE_NAME = "1.04. Real-life example.csv"
BUNDLE_PATH = "car_price_bundle.pkl"


def load_data() -> pd.DataFrame:
    """Load the raw dataset, preferring kagglehub, falling back to a local CSV."""
    # 1) Local file already sitting next to this script
    if os.path.exists(FILE_NAME):
        print(f"Loading local file: {FILE_NAME}")
        return pd.read_csv(FILE_NAME)

    # 2) kagglehub (same source used in the original notebook)
    try:
        import kagglehub
        from kagglehub import KaggleDatasetAdapter

        print("Downloading dataset via kagglehub ...")
        df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            "smritisingh1997/car-salescsv",
            FILE_NAME,
        )
        return df
    except Exception as e:
        raise RuntimeError(
            "Could not load the dataset via kagglehub and no local CSV was "
            f"found named '{FILE_NAME}'. Either configure Kaggle credentials "
            "(~/.kaggle/kaggle.json) or download the CSV manually from the "
            "Kaggle dataset 'smritisingh1997/car-salescsv' and place it next "
            f"to this script.\n\nOriginal error: {e}"
        )


def main():
    df = load_data()

    # ---- Cleaning (same steps as the notebook) ----
    df = df.dropna(subset=["Price", "EngineV"])
    df = df[df["EngineV"] <= 10]
    df["Log_price"] = np.log(df["Price"])
    df = df.drop(columns=["Model", "Price"])

    # ---- Encode categoricals, keep the encoders for the app ----
    encoders = {}
    cat_cols = df.select_dtypes(include="object").columns
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    X = df.drop(columns=["Log_price"])
    y = df["Log_price"]
    feature_order = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(),
        "Lasso Regression": Lasso(),
    }

    results = {}
    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        results[name] = {
            "r2": float(r2_score(y_test, preds)),
            "mae": float(mean_absolute_error(y_test, preds)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
        }
        trained[name] = model
        print(f"{name:20s}  R2={results[name]['r2']:.4f}  "
              f"MAE={results[name]['mae']:.4f}  RMSE={results[name]['rmse']:.4f}")

    best_name = max(results, key=lambda k: results[k]["r2"])
    print(f"\nBest model by R2: {best_name}")

    bundle = {
        "models": trained,
        "encoders": encoders,
        "feature_order": feature_order,
        "results": results,
        "best_model": best_name,
        "numeric_ranges": {
            "Mileage": (int(X["Mileage"].min()), int(X["Mileage"].max())),
            "EngineV": (float(X["EngineV"].min()), float(X["EngineV"].max())),
            "Year": (int(X["Year"].min()), int(X["Year"].max())),
        },
    }

    joblib.dump(bundle, BUNDLE_PATH)
    print(f"\nSaved bundle -> {BUNDLE_PATH}")


if __name__ == "__main__":
    main()
