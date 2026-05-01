"""
Микросервис инференса модели CatBoost для прогнозирования оттока.
Принимает JSON с 36 признаками, возвращает churn_probability и churn_prediction.
"""
from flask import Flask, request, jsonify
import pandas as pd
from catboost import CatBoostClassifier
import os

app = Flask(__name__)

# Путь к обученной модели (лежит в папке models проекта)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "catboost_churn_model.pkl")
THRESHOLD = 0.25          # оптимальный порог, подобранный на validation

# Загружаем модель при старте сервера
print("Loading CatBoost model...")
if not os.path.exists(MODEL_PATH):
    print(f"ERROR: Model file not found at {MODEL_PATH}")
    print("Please run 'python src/main.py' first to train the model.")
    model = None
    FEATURE_NAMES = []
else:
    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)
    FEATURE_NAMES = model.feature_names_
    print(f"Model loaded successfully. Expected features: {len(FEATURE_NAMES)}")


@app.route("/predict", methods=["POST"])
def predict():
    """Основной endpoint для получения прогноза."""
    if model is None:
        return jsonify({"error": "Model not loaded. Train the model first."}), 500

    data = request.get_json()
    if not data or "features" not in data:
        return jsonify({"error": "Request must contain 'features' object"}), 400

    features = data["features"]

    # Проверяем наличие всех необходимых признаков
    missing = [f for f in FEATURE_NAMES if f not in features]
    if missing:
        return jsonify({
            "error": f"Missing features: {missing}",
            "expected_count": len(FEATURE_NAMES),
            "provided_count": len(features)
        }), 400

    try:
        # Строим DataFrame из одного наблюдения (порядок колонок как в модели)
        input_df = pd.DataFrame([features])[FEATURE_NAMES]

        # Категориальные признаки должны быть строками
        for col in ["city", "gender", "preferred_payment"]:
            if col in input_df.columns:
                input_df[col] = input_df[col].astype(str)

        prob = float(model.predict_proba(input_df)[:, 1][0])
        pred = int(prob >= THRESHOLD)

        return jsonify({
            "churn_probability": round(prob, 6),
            "churn_prediction": pred,
            "threshold": THRESHOLD
        })
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route("/health", methods=["GET"])
def health():
    """Проверка работоспособности сервера."""
    return jsonify({
        "status": "ok" if model is not None else "model not loaded",
        "model_path": MODEL_PATH,
        "features_count": len(FEATURE_NAMES)
    })


@app.route("/features", methods=["GET"])
def get_features():
    """Возвращает список ожидаемых признаков."""
    return jsonify({
        "count": len(FEATURE_NAMES),
        "features": FEATURE_NAMES
    })


if __name__ == "__main__":
    print("=" * 60)
    print("Churn Prediction Inference Server")
    print("=" * 60)
    print(f"Model path : {MODEL_PATH}")
    print(f"Threshold  : {THRESHOLD}")
    print(f"Features   : {len(FEATURE_NAMES)}")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=False)