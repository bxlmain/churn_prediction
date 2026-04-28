import pandas as pd


def save_final_model_summary(
    metrics_path: str = "reports/model_metrics.csv",
    output_path: str = "reports/final_model_summary.csv",
):
    metrics = pd.read_csv(metrics_path)

    selected_rows = metrics[
        metrics["dataset"].isin(
            [
                "OOT RandomForest" "OOT CatBoost",
                "OOT CatBoost Tuned Threshold",
            ]
        )
    ].copy()

    selected_rows["model"] = selected_rows["dataset"].map(
        {
            "OOT RandomForest": "RandomForest baseline",
            "OOT CatBoost": "CatBoost threshold=0.5",
            "OOT CatBoost Tuned Threshold": "CatBoost threshold=0.25",
        }
    )

    columns_order = [
        "model",
        "dataset",
        "threshold",
        "roc_auc",
        "pr_auc",
        "precision",
        "recall",
        "f1",
    ]

    for col in columns_order:
        if col not in selected_rows.columns:
            selected_rows[col] = None

    selected_rows = selected_rows[columns_order]

    selected_rows.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Saved final model summary to {output_path}")

    return selected_rows
