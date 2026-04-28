import pandas as pd
import matplotlib.pyplot as plt


def save_feature_importance(
    model, feature_names, output_csv_path, output_png_path, top_n=20
):
    importance = model.get_feature_importance()

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importance,
        }
    )

    importance_df = importance_df.sort_values(
        by="importance", ascending=False
    ).reset_index(drop=True)

    importance_df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")

    top_features = importance_df.head(top_n).sort_values(
        by="importance", ascending=True
    )

    plt.figure(figsize=(10, 8))
    plt.barh(top_features["feature"], top_features["importance"])
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.title(f"Top {top_n} CatBoost Feature Importance")
    plt.tight_layout()
    plt.savefig(output_png_path, dpi=300)
    plt.close()

    print(f"Saved feature importance to {output_csv_path}")
    print(f"Saved feature importance plor to {output_png_path}")

    return importance_df
