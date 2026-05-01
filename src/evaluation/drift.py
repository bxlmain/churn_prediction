import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


EPS = 1e-6


def _safe_share(series: pd.Series) -> pd.Series:
    """
    Возвращает доли значений с защитой от нулей.
    """
    shares = series / series.sum()
    return shares.replace(0, EPS).fillna(EPS)


def calculate_numeric_psi(
    expected: pd.Series,
    actual: pd.Series,
    bins: int = 10,
) -> float:
    """
    Расчёт PSI для числового признака.

    expected — базовое распределение, обычно Train.
    actual — новое распределение, обычно OOT.
    """
    expected = pd.to_numeric(expected, errors="coerce").dropna()
    actual = pd.to_numeric(actual, errors="coerce").dropna()

    if expected.empty or actual.empty:
        return np.nan

    if expected.nunique() <= 1:
        return 0.0

    try:
        breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))

        breakpoints = np.unique(breakpoints)

        if len(breakpoints) <= 2:
            return 0.0

        expected_bins = pd.cut(expected, bins=breakpoints, include_lowest=True)

        actual_bins = pd.cut(actual, bins=breakpoints, include_lowest=True)

        expected_counts = expected_bins.value_counts(sort=False)
        actual_counts = actual_bins.value_counts(sort=False)

        expected_share = _safe_share(expected_counts)
        actual_share = _safe_share(actual_counts)

        psi_values = (actual_share - expected_share) * np.log(
            actual_share / expected_share
        )

        return float(psi_values.sum())

    except Exception:
        return np.nan


def calculate_categorical_psi(expected: pd.Series, actual: pd.Series) -> float:
    """
    Расчёт PSI для категориального признака.
    """
    expected = expected.fillna("unknown").astype(str)
    actual = actual.fillna("unknown").astype(str)

    expected_counts = expected.value_counts()
    actual_counts = actual.value_counts()

    all_categories = expected_counts.index.union(actual_counts.index)

    expected_counts = expected_counts.reindex(all_categories, fill_value=0)
    actual_counts = actual_counts.reindex(all_categories, fill_value=0)

    expected_share = _safe_share(expected_counts)
    actual_share = _safe_share(actual_counts)

    psi_values = (actual_share - expected_share) * np.log(actual_share / expected_share)

    return float(psi_values.sum())


def interpret_psi(psi_value: float) -> str:
    """
    Текстовая интерпретация PSI.
    """
    if pd.isna(psi_value):
        return "не рассчитано"

    if psi_value < 0.10:
        return "значимого сдвига нет"

    if psi_value < 0.25:
        return "умеренный сдвиг"

    return "существенный сдвиг"


def calculate_psi_report(
    train_df: pd.DataFrame,
    oot_df: pd.DataFrame,
    feature_columns: list[str],
    categorical_features: list[str],
    output_csv_path: str = "reports/psi_train_oot.csv",
    bins: int = 10,
) -> pd.DataFrame:
    """
    Формирует PSI-отчёт между Train и OOT.

    train_df — обучающая выборка;
    oot_df — out-of-time выборка;
    feature_columns — список признаков модели;
    categorical_features — список категориальных признаков.
    """
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    rows = []

    for feature in feature_columns:
        if feature not in train_df.columns or feature not in oot_df.columns:
            continue

        if feature in categorical_features:
            psi_value = calculate_categorical_psi(train_df[feature], oot_df[feature])
            feature_type = "categorical"
        else:
            psi_value = calculate_numeric_psi(
                train_df[feature], oot_df[feature], bins=bins
            )
            feature_type = "numeric"

        rows.append(
            {
                "feature": feature,
                "feature_type": feature_type,
                "psi": psi_value,
                "interpretation": interpret_psi(psi_value),
            }
        )

    psi_df = pd.DataFrame(rows)
    psi_df = psi_df.sort_values("psi", ascending=False)

    psi_df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")

    return psi_df


def save_psi_plot(
    psi_df: pd.DataFrame,
    output_png_path: str = "reports/psi_train_oot.png",
    top_n: int = 20,
) -> str:
    """
    Сохраняет график top-N признаков по PSI.
    """
    os.makedirs(os.path.dirname(output_png_path), exist_ok=True)

    plot_df = (
        psi_df.dropna(subset=["psi"])
        .sort_values("psi", ascending=False)
        .head(top_n)
        .sort_values("psi", ascending=True)
    )

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.barh(plot_df["feature"], plot_df["psi"])

    ax.axvline(0.10, linestyle="--", linewidth=1, label="PSI = 0.10")
    ax.axvline(0.25, linestyle="--", linewidth=1, label="PSI = 0.25")

    ax.set_title(f"Top {top_n} признаков по PSI между Train и OOT")
    ax.set_xlabel("PSI")
    ax.set_ylabel("Признак")
    ax.grid(True, axis="x", alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_png_path, dpi=300)
    plt.close(fig)

    return output_png_path
