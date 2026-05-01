from src.data.load_data import load_data
from src.features.build_features import build_features
from src.models.train_model import train_model


def main():
    print("Step 1: Loading data...")
    data = load_data()

    print("Step 2: Building features...")
    df_features = build_features(data)

    df_features.to_csv("data/ml_dataset.csv", index=False, encoding="utf-8-sig")
    print("Saved to data/ml_dataset.csv")

    print("Step 3: Training model...")
    model, evaluation_data = train_model(df_features, data["config"])

    print("\nBest CatBoost threshold:")
    print(evaluation_data["best_threshold"])

    print("\nValidation CatBoost tuned metrics:")
    print(evaluation_data["catboost_valid_tuned_metrics"])

    print("\nOOT CatBoost tuned metrics:")
    print(evaluation_data["catboost_oot_tuned_metrics"])

    print("\nTop PSI features:")
    print(evaluation_data["psi_report"].head(10))

    print("Pipeline finished successfully.")


if __name__ == "__main__":
    main()
