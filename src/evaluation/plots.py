import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    RocCurveDisplay,
    PrecisionRecallDisplay,
    ConfusionMatrixDisplay,
    confusion_matrix,
)


def ensure_dir(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)


def save_roc_curve(
    y_true, y_proba, dataset_name: str, output_dir: str = "reports"
) -> str:
    """
    Сохраняет ROC-кривую для указанной выборки.
    """
    ensure_dir(output_dir)

    fig, ax = plt.subplots(figsize=(7, 5))
    RocCurveDisplay.from_predictions(y_true, y_proba, ax=ax)

    ax.set_title(f"ROC-кривая ({dataset_name})")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.grid(True, alpha=0.3)

    file_path = os.path.join(output_dir, f"roc_curve_{dataset_name}.png")
    plt.tight_layout()
    plt.savefig(file_path, dpi=300)
    plt.close(fig)

    return file_path


def save_pr_curve(
    y_true, y_proba, dataset_name: str, output_dir: str = "reports"
) -> str:
    """
    Сохраняет Precision-Recall кривую для указанной выборки.
    """
    ensure_dir(output_dir)

    fig, ax = plt.subplots(figsize=(7, 5))
    PrecisionRecallDisplay.from_predictions(y_true, y_proba, ax=ax)

    ax.set_title(f"Precision-Recall кривая ({dataset_name})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.grid(True, alpha=0.3)

    file_path = os.path.join(output_dir, f"pr_curve_{dataset_name}.png")
    plt.tight_layout()
    plt.savefig(file_path, dpi=300)
    plt.close(fig)

    return file_path


def save_confusion_matrix_plot(
    y_true, y_pred, dataset_name: str, output_dir: str = "reports"
) -> str:
    """
    Сохраняет изображение матрицы ошибок.
    """
    ensure_dir(output_dir)

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["active", "churn"]
    )
    display.plot(ax=ax, values_format="d", colorbar=False)

    ax.set_title(f"Матрица ошибок ({dataset_name})")

    file_path = os.path.join(output_dir, f"confusion_matrix_{dataset_name}.png")
    plt.tight_layout()
    plt.savefig(file_path, dpi=300)
    plt.close(fig)

    return file_path


def save_confusion_matrix_table(
    y_true, y_pred, dataset_name: str, output_dir: str = "reports"
) -> str:
    """
    Сохраняет матрицу ошибок в CSV.
    """
    ensure_dir(output_dir)

    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=["actual_active", "actual_churn"],
        columns=["predicted_active", "predicted_churn"],
    )

    file_path = os.path.join(output_dir, f"confusion_matrix_{dataset_name}.csv")
    cm_df.to_csv(file_path, encoding="utf-8-sig")

    return file_path


def save_threshold_plot(
    threshold_df: pd.DataFrame,
    output_dir: str = "reports",
    file_name: str = "catboost_threshold_selection.png",
) -> str:
    """
    Строит график зависимости Precision, Recall и F1 от порога классификации.

    Ожидаемые колонки:
    threshold, precision, recall, f1
    """
    ensure_dir(output_dir)

    df = threshold_df.copy()

    if "f1_score" in df.columns and "f1" not in df.columns:
        df = df.rename(columns={"f1_score": "f1"})

    required_columns = {"threshold", "precision", "recall", "f1"}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Не хватает колонок для графика порога: {missing}")

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(df["threshold"], df["precision"], marker="o", label="Precision")
    ax.plot(df["threshold"], df["recall"], marker="o", label="Recall")
    ax.plot(df["threshold"], df["f1"], marker="o", label="F1-score")

    ax.set_title("Подбор порога классификации CatBoost")
    ax.set_xlabel("Порог классификации")
    ax.set_ylabel("Значение метрики")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()

    file_path = os.path.join(output_dir, file_name)
    plt.tight_layout()
    plt.savefig(file_path, dpi=300)
    plt.close(fig)

    return file_path


def save_metrics_comparison_plot(
    metrics_df: pd.DataFrame,
    output_dir: str = "reports",
    file_name: str = "model_metrics_comparison.png",
) -> str:
    """
    Строит столбчатый график сравнения моделей по основным метрикам.

    Ожидаемые колонки:
    dataset, roc_auc, pr_auc, precision, recall, f1
    """
    ensure_dir(output_dir)

    df = metrics_df.copy()

    required_columns = {"dataset", "roc_auc", "pr_auc", "precision", "recall", "f1"}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Не хватает колонок для сравнения метрик: {missing}")

    preferred_rows = [
        "Validation RandomForest",
        "OOT RandomForest",
        "Validation CatBoost",
        "OOT CatBoost",
        "Validation CatBoost Tuned Threshold",
        "OOT CatBoost Tuned Threshold",
    ]

    plot_df = df[df["dataset"].isin(preferred_rows)].copy()

    if plot_df.empty:
        plot_df = df.copy()

    plot_df = plot_df.set_index("dataset")[
        [
            "roc_auc",
            "pr_auc",
            "precision",
            "recall",
            "f1",
        ]
    ]

    fig, ax = plt.subplots(figsize=(11, 6))
    plot_df.plot(kind="bar", ax=ax)

    ax.set_title("Сравнение качества моделей")
    ax.set_xlabel("Выборка и модель")
    ax.set_ylabel("Значение метрики")
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(title="Метрика")

    plt.xticks(rotation=35, ha="right")

    file_path = os.path.join(output_dir, file_name)
    plt.tight_layout()
    plt.savefig(file_path, dpi=300)
    plt.close(fig)

    return file_path


def save_class_distribution_plot(
    train,
    valid,
    oot,
    output_dir: str = "reports",
    file_name: str = "class_distribution.png",
) -> str:
    """
    Сохраняет график распределения классов churn_flag по Train, Validation и OOT.
    """
    ensure_dir(output_dir)

    rows = []

    for name, df in [
        ("Train", train),
        ("Validation", valid),
        ("OOT", oot),
    ]:
        value_counts = df["churn_flag"].value_counts(normalize=True).to_dict()

        rows.append(
            {
                "dataset": name,
                "active_share": value_counts.get(0, 0),
                "churn_share": value_counts.get(1, 0),
            }
        )

    dist_df = pd.DataFrame(rows)
    dist_df.to_csv(
        os.path.join(output_dir, "class_distribution.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    plot_df = dist_df.set_index("dataset")[["active_share", "churn_share"]]

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_df.plot(kind="bar", stacked=True, ax=ax)

    ax.set_title("Распределение классов по выборкам")
    ax.set_xlabel("Выборка")
    ax.set_ylabel("Доля класса")
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(["active (0)", "churn (1)"])

    plt.xticks(rotation=0)

    file_path = os.path.join(output_dir, file_name)
    plt.tight_layout()
    plt.savefig(file_path, dpi=300)
    plt.close(fig)

    return file_path


def save_probability_distribution_plot(
    y_proba, y_true, dataset_name: str, output_dir: str = "reports"
) -> str:
    """
    Сохраняет распределение предсказанных вероятностей для классов active/churn.
    """
    ensure_dir(output_dir)

    df = pd.DataFrame(
        {
            "y_true": y_true,
            "y_proba": y_proba,
        }
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    df[df["y_true"] == 0]["y_proba"].plot(
        kind="hist", bins=25, alpha=0.6, ax=ax, label="actual active"
    )

    df[df["y_true"] == 1]["y_proba"].plot(
        kind="hist", bins=25, alpha=0.6, ax=ax, label="actual churn"
    )

    ax.set_title(f"Распределение вероятностей оттока ({dataset_name})")
    ax.set_xlabel("Предсказанная вероятность churn")
    ax.set_ylabel("Количество наблюдений")
    ax.grid(True, alpha=0.3)
    ax.legend()

    file_path = os.path.join(output_dir, f"probability_distribution_{dataset_name}.png")
    plt.tight_layout()
    plt.savefig(file_path, dpi=300)
    plt.close(fig)

    return file_path
