import os
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import joblib

COLUMN_NAMES = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country",
    "income",
]

NUMERIC_COLS = ["age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
CATEGORICAL_COLS = ["workclass", "education", "marital_status", "occupation", "relationship", "race", "sex", "native_country"]

def load_adult_csv(path):
    """Load one of the raw Adult CSV files into a clean DataFrame."""
    return pd.read_csv(
        path,
        names=COLUMN_NAMES,
        na_values="?",          # standard missing values marker
        skipinitialspace=True,  # strips leading spaces
        skiprows=1,             # skip the header row
        engine='python',
        on_bad_lines='skip'
    )

def main():
    print("Starting model training pipeline...")
    
    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(base_dir, "adult_income", "adult_train.csv")
    test_path = os.path.join(base_dir, "adult_income", "adult_test.csv")
    
    print(f"Loading train data from: {train_path}")
    print(f"Loading test data from: {test_path}")
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("Error: Train or test CSV files not found.")
        return
        
    train_df = load_adult_csv(train_path)
    test_df = load_adult_csv(test_path)
    
    # Combine into a single DataFrame for cleaning
    train_df["split"] = "train"
    test_df["split"] = "test"
    df = pd.concat([train_df, test_df], ignore_index=True)
    
    print(f"Loaded combined shape: {df.shape}")
    
    # Clean numeric columns: coerce invalid values (like the cross validator row) to NaN
    print("Casting numeric columns and cleaning strings...")
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    # Clean text columns: strip spaces and standardise
    for col in CATEGORICAL_COLS + ["income"]:
        df[col] = df[col].astype(str).str.strip()
        
    # Replace the text "nan" (if any resulted from casting/missing values) with actual NaN
    df = df.replace("nan", np.nan)
    df = df.replace("None", np.nan)
    
    # Clean income labels (removing trailing periods from the test set labels)
    df["income"] = df["income"].str.rstrip(".")
    
    # Drop rows with missing values
    rows_before_dropna = len(df)
    df = df.dropna(subset=NUMERIC_COLS + CATEGORICAL_COLS + ["income"])
    df = df.reset_index(drop=True)
    print(f"Dropped {rows_before_dropna - len(df)} rows with missing values.")
    
    # Drop duplicates
    rows_before_dups = len(df)
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)
    print(f"Dropped {rows_before_dups - len(df)} duplicate rows.")
    
    # Convert income to binary target (1 = >50K, 0 = <=50K)
    df["income_binary"] = (df["income"] == ">50K").astype(int)
    
    print(f"Cleaned data shape: {df.shape}")
    print("Income target distribution:")
    print(df["income_binary"].value_counts(normalize=True))
    
    # Separate features and target
    X = df[NUMERIC_COLS + CATEGORICAL_COLS]
    y = df["income_binary"]
    
    # Print data types
    print("\nFeature data types:")
    print(X.dtypes)
    
    # Build preprocessor
    # categorical values are one-hot encoded, numeric values passed through
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
            ("num", "passthrough", NUMERIC_COLS)
        ]
    )
    
    # XGBoost model configuration (from the notebook)
    xgb_model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42
    )
    
    # Build full pipeline
    model_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", xgb_model)
    ])
    
    # Fit the pipeline
    print("\nFitting model pipeline on the entire dataset...")
    model_pipeline.fit(X, y)
    
    # Save the pipeline
    model_output_path = os.path.join(base_dir, "model.joblib")
    print(f"Saving model to: {model_output_path}")
    joblib.dump(model_pipeline, model_output_path)
    
    # Save the list of categorical feature values for the web app UI
    # This allows us to populate the frontend select inputs dynamically.
    categorical_categories = {}
    for col in CATEGORICAL_COLS:
        # Sort and filter unique values
        unique_vals = sorted([str(val) for val in df[col].unique()])
        categorical_categories[col] = unique_vals
        
    categories_json_path = os.path.join(base_dir, "categories.joblib")
    print(f"Saving categorical mappings to: {categories_json_path}")
    joblib.dump(categorical_categories, categories_json_path)
    
    print("\nModel training pipeline completed successfully!")

if __name__ == "__main__":
    main()
