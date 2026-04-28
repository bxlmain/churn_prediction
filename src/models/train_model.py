import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
import joblib

from src.evaluation.feature_importance import save_feature_importance
from src.evaluation.final_summary import save_final_model_summary


def split_by_time(df: pd.DataFrame, config: dict):
    df = df.copy()
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])

    train_start = pd.to_datetime(config["split"]["train_start"])
    train_end = pd.to_datetime(config["split"]["train_end"])
    valid_start = pd.to_datetime(config["split"]["valid_start"])
    valid_end = pd.to_datetime(config["split"]["valid_end"])
    oot_date = pd.to_datetime(config["split"]["oot_date"])

    train = df[
        (df["snapshot_date"] >= train_start) & (df["snapshot_date"] <= train_end)
    ].copy()
    valid = df[
        (df["snapshot_date"] >= valid_start) & (df["snapshot_date"] <= valid_end)
    ].copy()

    oot = df[df["snapshot_date"] == oot_date].copy()

    return train, valid, oot


def prepare_xy(df: pd.DataFrame):
    drop_cols = [
        "customer_id",
        "registration_date",
        "snapshot_date",
        "churn_flag",
    ]
    X = df.drop(columns=drop_cols)
    y = df["churn_flag"]

    cat_features = X.select_dtypes(include=["object"]).columns.tolist()
    return X, y, cat_features


def prepare_xy_for_random_forest(
    train: pd.DataFrame, valid: pd.DataFrame, oot: pd.DataFrame
):
    X_train, y_train, _ = prepare_xy(train)
    X_valid, y_valid, _ = prepare_xy(valid)
    X_oot, y_oot, _ = prepare_xy(oot)

    X_train_rf = pd.get_dummies(X_train, drop_first=False)
    X_valid_rf = pd.get_dummies(X_valid, drop_first=False)
    X_oot_rf = pd.get_dummies(X_oot, drop_first=False)

    X_valid_rf = X_valid_rf.reindex(columns=X_train_rf.columns, fill_value=0)
    X_oot_rf = X_oot_rf.reindex(columns=X_train_rf.columns, fill_value=0)

    return X_train_rf, y_train, X_valid_rf, y_valid, X_oot_rf, y_oot


def evaluate_model(model, X, y, dataset_name: str):
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    metrics = {
        "dataset": dataset_name,
        "roc_auc": roc_auc_score(y, y_proba),
        "pr_auc": average_precision_score(y, y_proba),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
    }

    print(f"\n==== {dataset_name} metrics ====")
    for key, value in metrics.items():
        if key != "dataset":
            print(f"{key}: {value:.4f}")

    print("\nConfusion matrix:")
    print(confusion_matrix(y, y_pred))

    print("\nClassification report:")
    print(classification_report(y, y_pred, zero_division=0))

    return metrics


def find_best_threshold(model, X_valid, y_valid):
    y_proba = model.predict_proba(X_valid)[:, 1]

    thresholds = [i / 100 for i in range(10, 91, 5)]
    rows = []

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)

        rows.append(
            {
                "threshold": threshold,
                "precision": precision_score(y_valid, y_pred, zero_division=0),
                "recall": recall_score(y_valid, y_pred, zero_division=0),
                "f1": f1_score(y_valid, y_pred, zero_division=0),
            }
        )

    threshold_df = pd.DataFrame(rows)

    best_row = threshold_df.sort_values("f1", ascending=False).iloc[0]
    best_threshold = best_row["threshold"]

    print("\nThreshold selection on validation:")
    print(threshold_df)

    print(
        f"\nBest threshold by F1: {best_threshold:.2f} "
        f"(precision={best_row['precision']:.4f}, "
        f"recall={best_row['recall']:.4f}, "
        f"f1={best_row['f1']:.4f})"
    )

    return best_threshold, threshold_df


def evaluate_model_with_threshold(model, X, y, dataset_name: str, threshold: float):
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    metrics = {
        "dataset": dataset_name,
        "threshold": threshold,
        "roc_auc": roc_auc_score(y, y_proba),
        "pr_auc": average_precision_score(y, y_proba),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
    }

    print(f"\n===== {dataset_name} metrics with threshold={threshold:.2f} =====")
    for key, value in metrics.items():
        if key not in ["dataset"]:
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")

    print("\nConfusion matrix:")
    print(confusion_matrix(y, y_pred))

    print("\nClassification report:")
    print(classification_report(y, y_pred, zero_division=0))

    return metrics


def train_model(df: pd.DataFrame, config: dict):
    print("Splitting dataset by time...")

    train, valid, oot = split_by_time(df, config)

    print(f"Train shape: {train.shape}")
    print(f"Validation shape: {valid.shape}")
    print(f"OOT shape: {oot.shape}")

    X_train, y_train, cat_features = prepare_xy(train)

    pd.DataFrame({"feature": X_train.columns.tolist()}).to_csv(
        "reports/model_features.csv",
        index=False,
        encoding="utf-8-sig",
    )

    X_valid, y_valid, _ = prepare_xy(valid)
    X_oot, y_oot, _ = prepare_xy(oot)

    X_train_rf, y_train_rf, X_valid_rf, y_valid_rf, X_oot_rf, y_oot_rf = (
        prepare_xy_for_random_forest(train, valid, oot)
    )

    print(f"Categorical features for CatBoost: {cat_features}")

    print("\nTraining baseline Randomforest...")
    baseline = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        random_state=config["model"]["random_state"],
        class_weight="balanced",
        n_jobs=-1,
    )

    baseline.fit(X_train_rf, y_train_rf)

    baseline_train_metrics = evaluate_model(
        baseline, X_train_rf, y_train_rf, "Train RandomForest"
    )

    baseline_valid_metrics = evaluate_model(
        baseline, X_valid_rf, y_valid_rf, "Validation RandomForest"
    )

    baseline_oot_metrics = evaluate_model(
        baseline, X_oot_rf, y_oot_rf, "OOT RandomForest"
    )

    print("\nTraining CatBoostClassifier...")

    catboost = CatBoostClassifier(
        iterations=config["model"]["catboost_iterations"],
        learning_rate=config["model"]["catboost_learning_rate"],
        depth=config["model"]["catboost_depth"],
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=config["model"]["random_state"],
        verbose=100,
    )

    catboost.fit(
        X_train,
        y_train,
        cat_features=cat_features,
        eval_set=(X_valid, y_valid),
        use_best_model=True,
    )

    catboost_valid_metrics = evaluate_model(
        catboost, X_valid, y_valid, "Validation CatBoost"
    )

    catboost_oot_metrics = evaluate_model(catboost, X_oot, y_oot, "OOT CatBoost")

    best_threshold, threshold_df = find_best_threshold(catboost, X_valid, y_valid)

    catboost_valid_tuned_metrics = evaluate_model_with_threshold(
        catboost,
        X_valid,
        y_valid,
        "Validation CatBoost Tuned Threshold",
        best_threshold,
    )

    catboost_oot_tuned_metrics = evaluate_model_with_threshold(
        catboost,
        X_oot,
        y_oot,
        "OOT CatBoost Tuned Threshold",
        best_threshold,
    )

    threshold_df.to_csv(
        "reports/catboost_threshold_selection.csv",
        index=False,
        encoding="utf-8-sig",
    )

    feature_importance_df = save_feature_importance(
        model=catboost,
        feature_names=X_train.columns.tolist(),
        output_csv_path="reports/catboost_feature_importance.csv",
        output_png_path="reports/catboost_feature_importance.png",
        top_n=20,
    )

    joblib.dump(baseline, "models/random_forest_baseline.pkl")
    joblib.dump(catboost, "models/catboost_churn_model.pkl")

    metrics_df = pd.DataFrame(
        [
            baseline_train_metrics,
            baseline_valid_metrics,
            baseline_oot_metrics,
            catboost_valid_metrics,
            catboost_oot_metrics,
            catboost_valid_tuned_metrics,
            catboost_oot_tuned_metrics,
        ]
    )

    metrics_df.to_csv("reports/model_metrics.csv", index=False, encoding="utf-8-sig")

    save_final_model_summary(
        metrics_path="reports/model_metrics.csv",
        output_path="reports/final_model_summary.csv",
    )

    print("\nSaved baseline to models/random_forest_baseline.pkl")
    print("Saved model to models/catboost_churn_model.pkl")
    print("Saved metrics to reports/model_metrics.csv")

    return catboost
