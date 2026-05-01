import os
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
from src.evaluation.plots import (
    save_roc_curve,
    save_pr_curve,
    save_confusion_matrix_plot,
    save_confusion_matrix_table,
    save_threshold_plot,
    save_metrics_comparison_plot,
    save_class_distribution_plot,
    save_probability_distribution_plot,
)
from src.evaluation.drift import calculate_psi_report, save_psi_plot


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


def evaluate_model(model, X, y, dataset_name: str, threshold: float = 0.5):
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

    print(f"\n==== {dataset_name} metrics, threshold={threshold:.2f} ====")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    print("\nConfusion matrix:")
    print(confusion_matrix(y, y_pred))

    print("\nClassification report:")
    print(classification_report(y, y_pred, zero_division=0))

    return metrics, y_proba, y_pred


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
    best_threshold = float(best_row["threshold"])

    print("\nThreshold selection on validation:")
    print(threshold_df)

    print(
        f"\nBest threshold by F1: {best_threshold:.2f} "
        f"(precision={best_row['precision']:.4f}, "
        f"recall={best_row['recall']:.4f}, "
        f"f1={best_row['f1']:.4f})"
    )

    return best_threshold, threshold_df


def save_prediction_table(
    customer_ids,
    snapshot_dates,
    y_true,
    y_proba,
    y_pred,
    dataset_name: str,
    output_dir: str = "reports",
):
    os.makedirs(output_dir, exist_ok=True)

    pred_df = pd.DataFrame(
        {
            "customer_id": customer_ids.values,
            "snapshot_date": snapshot_dates.values,
            "y_true": y_true.values,
            "churn_probability": y_proba,
            "churn_prediction": y_pred,
        }
    )

    output_path = os.path.join(output_dir, f"predictions_{dataset_name}.csv")
    pred_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    return output_path


def train_model(df: pd.DataFrame, config: dict):
    os.makedirs("reports", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    print("Splitting dataset by time...")

    train, valid, oot = split_by_time(df, config)

    print(f"Train shape: {train.shape}")
    print(f"Validation shape: {valid.shape}")
    print(f"OOT shape: {oot.shape}")

    save_class_distribution_plot(
        train=train,
        valid=valid,
        oot=oot,
        output_dir="reports",
        file_name="class_distribution.png",
    )

    X_train, y_train, cat_features = prepare_xy(train)
    X_valid, y_valid, _ = prepare_xy(valid)
    X_oot, y_oot, _ = prepare_xy(oot)

    print("\nCalculating PSI between Train and OOT...")

    psi_df = calculate_psi_report(
        train_df=X_train,
        oot_df=X_oot,
        feature_columns=X_train.columns.tolist(),
        categorical_features=cat_features,
        output_csv_path="reports/psi_train_oot.csv",
        bins=10,
    )

    save_psi_plot(
        psi_df=psi_df,
        output_png_path="reports/psi_train_oot.png",
        top_n=20,
    )

    print("Saved PSI report to reports/psi_train_oot.csv")
    print("Saved PSI plot to reports/psi_train_oot.png")

    pd.DataFrame({"feature": X_train.columns.tolist()}).to_csv(
        "reports/model_features.csv",
        index=False,
        encoding="utf-8-sig",
    )

    X_train_rf, y_train_rf, X_valid_rf, y_valid_rf, X_oot_rf, y_oot_rf = (
        prepare_xy_for_random_forest(train, valid, oot)
    )

    print(f"Categorical features for CatBoost: {cat_features}")

    print("\nTraining baseline RandomForest...")

    baseline = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        random_state=config["model"]["random_state"],
        class_weight="balanced",
        n_jobs=-1,
    )

    baseline.fit(X_train_rf, y_train_rf)

    baseline_train_metrics, _, _ = evaluate_model(
        baseline,
        X_train_rf,
        y_train_rf,
        "Train RandomForest",
        threshold=0.5,
    )

    baseline_valid_metrics, _, _ = evaluate_model(
        baseline,
        X_valid_rf,
        y_valid_rf,
        "Validation RandomForest",
        threshold=0.5,
    )

    baseline_oot_metrics, _, _ = evaluate_model(
        baseline,
        X_oot_rf,
        y_oot_rf,
        "OOT RandomForest",
        threshold=0.5,
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

    catboost_train_metrics, catboost_train_proba, catboost_train_pred = evaluate_model(
        catboost,
        X_train,
        y_train,
        "Train CatBoost",
        threshold=0.5,
    )

    catboost_valid_metrics, catboost_valid_proba, catboost_valid_pred = evaluate_model(
        catboost,
        X_valid,
        y_valid,
        "Validation CatBoost",
        threshold=0.5,
    )

    catboost_oot_metrics, catboost_oot_proba, catboost_oot_pred = evaluate_model(
        catboost,
        X_oot,
        y_oot,
        "OOT CatBoost",
        threshold=0.5,
    )

    best_threshold, threshold_df = find_best_threshold(catboost, X_valid, y_valid)

    (
        catboost_valid_tuned_metrics,
        catboost_valid_tuned_proba,
        catboost_valid_tuned_pred,
    ) = evaluate_model(
        catboost,
        X_valid,
        y_valid,
        "Validation CatBoost Tuned Threshold",
        threshold=best_threshold,
    )

    catboost_oot_tuned_metrics, catboost_oot_tuned_proba, catboost_oot_tuned_pred = (
        evaluate_model(
            catboost,
            X_oot,
            y_oot,
            "OOT CatBoost Tuned Threshold",
            threshold=best_threshold,
        )
    )

    threshold_df.to_csv(
        "reports/catboost_threshold_selection.csv",
        index=False,
        encoding="utf-8-sig",
    )

    save_threshold_plot(
        threshold_df=threshold_df,
        output_dir="reports",
        file_name="catboost_threshold_selection.png",
    )

    # ROC и PR-кривые строятся по вероятностям, поэтому порог здесь не нужен.
    save_roc_curve(
        y_true=y_valid,
        y_proba=catboost_valid_proba,
        dataset_name="validation",
        output_dir="reports",
    )

    save_roc_curve(
        y_true=y_oot,
        y_proba=catboost_oot_proba,
        dataset_name="oot",
        output_dir="reports",
    )

    save_pr_curve(
        y_true=y_valid,
        y_proba=catboost_valid_proba,
        dataset_name="validation",
        output_dir="reports",
    )

    save_pr_curve(
        y_true=y_oot,
        y_proba=catboost_oot_proba,
        dataset_name="oot",
        output_dir="reports",
    )

    save_probability_distribution_plot(
        y_proba=catboost_valid_proba,
        y_true=y_valid,
        dataset_name="validation",
        output_dir="reports",
    )

    save_probability_distribution_plot(
        y_proba=catboost_oot_proba,
        y_true=y_oot,
        dataset_name="oot",
        output_dir="reports",
    )

    # Матрицы ошибок при стандартном пороге 0.5.
    save_confusion_matrix_plot(
        y_true=y_valid,
        y_pred=catboost_valid_pred,
        dataset_name="validation_default",
        output_dir="reports",
    )

    save_confusion_matrix_plot(
        y_true=y_oot,
        y_pred=catboost_oot_pred,
        dataset_name="oot_default",
        output_dir="reports",
    )

    save_confusion_matrix_table(
        y_true=y_valid,
        y_pred=catboost_valid_pred,
        dataset_name="validation_default",
        output_dir="reports",
    )

    save_confusion_matrix_table(
        y_true=y_oot,
        y_pred=catboost_oot_pred,
        dataset_name="oot_default",
        output_dir="reports",
    )

    # Матрицы ошибок при подобранном пороге.
    save_confusion_matrix_plot(
        y_true=y_valid,
        y_pred=catboost_valid_tuned_pred,
        dataset_name="validation_tuned",
        output_dir="reports",
    )

    save_confusion_matrix_plot(
        y_true=y_oot,
        y_pred=catboost_oot_tuned_pred,
        dataset_name="oot_tuned",
        output_dir="reports",
    )

    save_confusion_matrix_table(
        y_true=y_valid,
        y_pred=catboost_valid_tuned_pred,
        dataset_name="validation_tuned",
        output_dir="reports",
    )

    save_confusion_matrix_table(
        y_true=y_oot,
        y_pred=catboost_oot_tuned_pred,
        dataset_name="oot_tuned",
        output_dir="reports",
    )

    save_prediction_table(
        customer_ids=valid["customer_id"],
        snapshot_dates=valid["snapshot_date"],
        y_true=y_valid,
        y_proba=catboost_valid_tuned_proba,
        y_pred=catboost_valid_tuned_pred,
        dataset_name="validation",
        output_dir="reports",
    )

    save_prediction_table(
        customer_ids=oot["customer_id"],
        snapshot_dates=oot["snapshot_date"],
        y_true=y_oot,
        y_proba=catboost_oot_tuned_proba,
        y_pred=catboost_oot_tuned_pred,
        dataset_name="oot",
        output_dir="reports",
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
            catboost_train_metrics,
            catboost_valid_metrics,
            catboost_oot_metrics,
            catboost_valid_tuned_metrics,
            catboost_oot_tuned_metrics,
        ]
    )

    metrics_df.to_csv(
        "reports/model_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    save_metrics_comparison_plot(
        metrics_df=metrics_df,
        output_dir="reports",
        file_name="model_metrics_comparison.png",
    )

    save_final_model_summary(
        metrics_path="reports/model_metrics.csv",
        output_path="reports/final_model_summary.csv",
    )

    evaluation_data = {
        "best_threshold": best_threshold,
        "catboost_valid_metrics": catboost_valid_metrics,
        "catboost_oot_metrics": catboost_oot_metrics,
        "catboost_valid_tuned_metrics": catboost_valid_tuned_metrics,
        "catboost_oot_tuned_metrics": catboost_oot_tuned_metrics,
        "feature_importance": feature_importance_df,
        "psi_report": psi_df,
    }

    print("\nSaved baseline to models/random_forest_baseline.pkl")
    print("Saved model to models/catboost_churn_model.pkl")
    print("Saved metrics to reports/model_metrics.csv")
    print("Saved plots and additional tables to reports/")

    return catboost, evaluation_data
