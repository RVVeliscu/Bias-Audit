import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from xgboost import XGBClassifier
import joblib
import pickle as pkl
from fairlearn.metrics import (
    MetricFrame,
    selection_rate,
    true_positive_rate,
    false_positive_rate,
    demographic_parity_difference,
    equalized_odds_difference,
)

RANDOM_STATE = 42

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.joblib")
FOREST_PATH = os.path.join(BASE_DIR, "best_random_forest_model.pkl")
CATEGORIES_PATH = os.path.join(BASE_DIR, "categories.joblib")

# Where the results get saved -- inside static/ so Flask can serve them
# directly as files (e.g. GET /metrics/fairness_report.json) without any
# extra route code.
METRICS_DIR = os.path.join(BASE_DIR, "static", "metrics")

COLUMN_NAMES = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country",
    "income",
]

NUMERIC_COLS = ["age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
CATEGORICAL_COLS = ["workclass", "education", "marital_status", "occupation", "relationship", "race", "sex", "native_country"]

# Metrics we want Fairlearn to break down by group. Together these show how
# often each model is right, how often it predicts ">50K", and where its
# errors land for each sensitive group.
FAIRNESS_SCORE_FUNCTIONS = {
    "accuracy": accuracy_score,
    "selection_rate": selection_rate,
    "true_positive_rate": true_positive_rate,
    "false_positive_rate": false_positive_rate,
}


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


def overall_metrics(y_true, y_pred, y_proba):
    """Headline performance metrics for one model, no group breakdown."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def audit_by_group(y_true, y_pred, sensitive_values):
    """Fairlearn breakdown for one model x one sensitive feature.

    Returns a plain dict (JSON-friendly) with the per-group metrics plus
    the two headline disparity metrics: demographic parity difference and
    equalized odds difference. Both are 0 when groups are treated
    identically, larger when there's a bigger gap between groups.
    """
    metric_frame = MetricFrame(
        metrics=FAIRNESS_SCORE_FUNCTIONS,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_values,
    )

    return {
        "by_group": metric_frame.by_group.to_dict(orient="index"),
        "overall": metric_frame.overall.to_dict(),
        "demographic_parity_difference": demographic_parity_difference(
            y_true, y_pred, sensitive_features=sensitive_values
        ),
        "equalized_odds_difference": equalized_odds_difference(
            y_true, y_pred, sensitive_features=sensitive_values
        ),
    }


def make_json_safe(obj):
    """Recursively convert numpy/pandas scalar types to plain Python types
    so json.dump doesn't choke on them."""
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return make_json_safe(obj.tolist())
    return obj


def save_overall_chart(overall, output_dir):
    """Grouped bar chart: accuracy/precision/recall/f1/roc_auc, XGBoost vs RF."""
    metric_names = list(next(iter(overall.values())).keys())
    rows = []
    for model_name, metrics in overall.items():
        for metric_name in metric_names:
            rows.append({"model": model_name, "metric": metric_name, "value": metrics[metric_name]})
    chart_df = pd.DataFrame(rows)

    plt.figure(figsize=(8, 5))
    sns.barplot(data=chart_df, x="metric", y="value", hue="model")
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.xlabel("")
    plt.title("Overall performance: XGBoost vs. Random Forest")
    plt.legend(title="Model")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "overall_comparison.png"), dpi=150)
    plt.close()


def save_group_chart(fairness_report, group_name, output_dir):
    """2x2 grid: one subplot per fairness metric, bars grouped by
    (sensitive group) x (model), for a single sensitive feature."""
    rows = []
    for model_name, groups in fairness_report.items():
        by_group = groups[group_name]["by_group"]
        for group_value, metrics in by_group.items():
            for metric_name, value in metrics.items():
                rows.append({
                    "model": model_name,
                    group_name: group_value,
                    "metric": metric_name,
                    "value": value,
                })
    chart_df = pd.DataFrame(rows)

    metric_names = list(FAIRNESS_SCORE_FUNCTIONS.keys())
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.flatten()

    for ax, metric_name in zip(axes, metric_names):
        subset = chart_df[chart_df["metric"] == metric_name]
        sns.barplot(data=subset, x=group_name, y="value", hue="model", ax=ax)
        ax.set_title(metric_name)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=30)
        ax.legend(title="Model", fontsize=8)

    fig.suptitle(f"Model comparison by {group_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{group_name}_breakdown.png"), dpi=150)
    plt.close()


def save_fairness_gap_chart(fairness_report, output_dir):
    """Bar chart comparing the headline disparity metrics (demographic
    parity difference, equalized odds difference) across models and
    sensitive features. Smaller bars = more equal treatment across groups."""
    rows = []
    for model_name, groups in fairness_report.items():
        for group_name, result in groups.items():
            rows.append({
                "model": model_name,
                "metric": f"Demographic parity diff. ({group_name})",
                "value": result["demographic_parity_difference"],
            })
            rows.append({
                "model": model_name,
                "metric": f"Equalized odds diff. ({group_name})",
                "value": result["equalized_odds_difference"],
            })
    chart_df = pd.DataFrame(rows)

    plt.figure(figsize=(9, 6))
    sns.barplot(data=chart_df, x="metric", y="value", hue="model")
    plt.ylabel("Disparity (0 = equal across groups)")
    plt.xlabel("")
    plt.xticks(rotation=20, ha="right")
    plt.title("Fairness gaps: XGBoost vs. Random Forest")
    plt.legend(title="Model", loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fairness_gaps.png"), dpi=150)
    plt.close()


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

    sensitive_features = df[["sex", "race"]]

    model = joblib.load(MODEL_PATH)
    forest_model = pkl.load(open(FOREST_PATH, "rb"))
    categories = joblib.load(CATEGORIES_PATH)

    # NOTE: this re-creates the exact train/test split used when the models
    # were trained (same random_state, test_size, and stratify). If either
    # model was actually trained with different split settings, these
    # "test" rows may overlap with what that model already saw during
    # training, which would make its metrics look better than they really
    # are. Keep this in sync with whatever split each training script used.
    X_train, X_test, y_train, y_test, sensitive_train, sensitive_test = train_test_split(
        X, y, sensitive_features,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,  # keep the same class balance in train and test
    )

    sex_test = sensitive_test["sex"]
    race_test = sensitive_test["race"]

    print(f"\nEvaluating on held-out test set: {X_test.shape[0]} rows")

    # ------------------------------------------------------------------
    # Run both models on the same held-out test set
    # ------------------------------------------------------------------
    print("\nRunning predictions...")

    xgb_pred = model.predict(X_test)
    xgb_proba = model.predict_proba(X_test)[:, 1]

    rf_pred = forest_model.predict(X_test)
    rf_proba = forest_model.predict_proba(X_test)[:, 1]

    # ------------------------------------------------------------------
    # Overall metrics (no group breakdown)
    # ------------------------------------------------------------------
    overall = {
        "xgboost": overall_metrics(y_test, xgb_pred, xgb_proba),
        "random_forest": overall_metrics(y_test, rf_pred, rf_proba),
    }

    print("\nOverall metrics:")
    print(json.dumps(make_json_safe(overall), indent=2))

    # ------------------------------------------------------------------
    # Fairlearn breakdown by sex and by race, for each model
    # ------------------------------------------------------------------
    print("\nRunning Fairlearn audit by sex and race...")

    fairness_report = {
        "xgboost": {
            "sex": audit_by_group(y_test, xgb_pred, sex_test),
            "race": audit_by_group(y_test, xgb_pred, race_test),
        },
        "random_forest": {
            "sex": audit_by_group(y_test, rf_pred, sex_test),
            "race": audit_by_group(y_test, rf_pred, race_test),
        },
    }

    # ------------------------------------------------------------------
    # Save everything to static/metrics/ so the Flask app can serve it
    # ------------------------------------------------------------------
    os.makedirs(METRICS_DIR, exist_ok=True)

    report = {
        "test_set_size": int(len(X_test)),
        "overall_metrics": overall,
        "fairness_by_group": fairness_report,
    }
    report = make_json_safe(report)

    report_path = os.path.join(METRICS_DIR, "fairness_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nSaved metrics report to {report_path}")

    # ------------------------------------------------------------------
    # Charts for the front end
    # ------------------------------------------------------------------
    sns.set_theme(style="whitegrid")
    save_overall_chart(overall, METRICS_DIR)
    save_group_chart(fairness_report, "sex", METRICS_DIR)
    save_group_chart(fairness_report, "race", METRICS_DIR)
    save_fairness_gap_chart(fairness_report, METRICS_DIR)

    print(f"Saved comparison charts to {METRICS_DIR}")
    print("\nDone.")


if __name__ == "__main__":
    main()