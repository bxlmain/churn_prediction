import joblib
import pandas as pd

MODEL_PATH = "models/catboost_churn_model.pkl"
DEFAULT_THRESHOLD = 0.25


def load_model(model_path: str = MODEL_PATH):
    return joblib.load(model_path)


def predict_churn(
    input_data: pd.DataFrame,
    model_path: str = MODEL_PATH,
    threshold: float = DEFAULT_THRESHOLD,
) -> pd.DataFrame:
    model = load.model(model_path)

    churn_probability = model.predict_proba(input_data)[:, 1]
    churn_prediction = (churn_probability >= threshold).astype(int)

    result = input_data.copy()
    result["churn_probability"] = churn_probability
    result["churn_prediction"] = churn_prediction

    return result
