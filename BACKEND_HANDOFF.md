2. Где лежит модель

Основная модель:

models/catboost_churn_model.pkl

Baseline-модель:

models/random_forest_baseline.pkl

Для backend-интеграции использовать:

models/catboost_churn_model.pkl
3. Рекомендуемый threshold

Финальный порог классификации:

0.25

Правило:

если churn_probability >= 0.25 → churn_prediction = 1
если churn_probability < 0.25  → churn_prediction = 0

Порог 0.25 был подобран на validation-выборке по максимальному F1-score и затем проверен на OOT-выборке.

4. Какие файлы важны для backend

Backend-разработчику нужны следующие файлы:

models/catboost_churn_model.pkl
reports/model_features.csv
src/models/predict.py
requirements.txt
MODEL_CONTRACT.md

Дополнительно для понимания качества модели:

reports/model_metrics.csv
reports/final_model_summary.csv
reports/catboost_threshold_selection.csv
reports/catboost_feature_importance.csv
reports/catboost_feature_importance.png
5. Входной формат модели

Модель принимает на вход pandas.DataFrame.

Формат:

одна строка = один клиент на одну дату прогнозирования

Количество входных признаков:

36

Список признаков хранится в файле:

reports/model_features.csv
6. Полный список признаков модели

Модель ожидает следующие признаки:

city
age
gender
preferred_payment
customer_lifetime_days
orders_total
orders_amount_total
orders_amount_mean
orders_quantity_total
orders_cancelled_total
delivered_orders_total
delivered_amount_total
delivered_amount_mean
delivered_orders_30d
delivered_amount_30d
delivered_orders_90d
delivered_amount_90d
days_since_last_order
visits_total
pages_viewed_total
pages_viewed_mean
cart_adds_total
checkout_initiated_total
session_duration_mean
visits_30d
cart_adds_30d
checkout_initiated_30d
visits_90d
cart_adds_90d
checkout_initiated_90d
days_since_last_visit
support_tickets_total
support_rating_mean
support_resolution_days_mean
support_tickets_90d
days_since_last_ticket
7. Категориальные признаки

CatBoost принимает категориальные признаки напрямую. Их не нужно кодировать вручную.

Категориальные признаки:

city
gender
preferred_payment

Они должны передаваться как строки.

Пример:

{
  "city": "Уфа",
  "gender": "Ж",
  "preferred_payment": "card"
}
8. Числовые признаки

Все остальные признаки должны быть числовыми (int или float).

Пример:

{
  "age": 35,
  "customer_lifetime_days": 420,
  "orders_total": 10,
  "orders_amount_total": 50000,
  "days_since_last_order": 60
}
9. Какие поля НЕ передаются в модель

Следующие поля являются служебными и не должны подаваться в модель:

customer_id
registration_date
snapshot_date
churn_flag

Назначение этих полей:

customer_id — идентификатор клиента
registration_date — дата регистрации клиента
snapshot_date — дата прогнозирования
churn_flag — целевая переменная, используется только при обучении
10. Как получить прогноз

В проекте должен использоваться файл:

src/models/predict.py

Рекомендуемое содержимое файла:

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
    model = load_model(model_path)

    churn_probability = model.predict_proba(input_data)[:, 1]
    churn_prediction = (churn_probability >= threshold).astype(int)

    result = input_data.copy()
    result["churn_probability"] = churn_probability
    result["churn_prediction"] = churn_prediction

    return result
11. Пример использования модели
import pandas as pd
from src.models.predict import predict_churn


input_data = pd.DataFrame(
    [
        {
            "city": "Уфа",
            "age": 35,
            "gender": "Ж",
            "preferred_payment": "card",
            "customer_lifetime_days": 420,
            "orders_total": 10,
            "orders_amount_total": 50000,
            "orders_amount_mean": 5000,
            "orders_quantity_total": 15,
            "orders_cancelled_total": 1,
            "delivered_orders_total": 9,
            "delivered_amount_total": 45000,
            "delivered_amount_mean": 5000,
            "delivered_orders_30d": 0,
            "delivered_amount_30d": 0,
            "delivered_orders_90d": 1,
            "delivered_amount_90d": 5000,
            "days_since_last_order": 60,
            "visits_total": 40,
            "pages_viewed_total": 300,
            "pages_viewed_mean": 7.5,
            "cart_adds_total": 12,
            "checkout_initiated_total": 5,
            "session_duration_mean": 180,
            "visits_30d": 3,
            "cart_adds_30d": 1,
            "checkout_initiated_30d": 0,
            "visits_90d": 8,
            "cart_adds_90d": 3,
            "checkout_initiated_90d": 1,
            "days_since_last_visit": 7,
            "support_tickets_total": 1,
            "support_rating_mean": 4.0,
            "support_resolution_days_mean": 2.0,
            "support_tickets_90d": 0,
            "days_since_last_ticket": 180,
        }
    ]
)

result = predict_churn(input_data)

print(result[["churn_probability", "churn_prediction"]])

Пример результата:

   churn_probability  churn_prediction
0           0.318742                 1

Так как 0.318742 >= 0.25, клиент классифицируется как склонный к оттоку.

12. Проверка состава признаков перед прогнозом

Перед вызовом модели необходимо проверить, что входной DataFrame содержит все признаки из reports/model_features.csv.

Рекомендуемый код:

import pandas as pd


features = pd.read_csv("reports/model_features.csv")["feature"].tolist()

missing_features = set(features) - set(input_data.columns)
extra_features = set(input_data.columns) - set(features)

if missing_features:
    raise ValueError(f"Missing features: {missing_features}")

if extra_features:
    input_data = input_data.drop(columns=list(extra_features))

input_data = input_data[features]
13. Важное требование к порядку колонок

Перед прогнозом нужно привести DataFrame к порядку колонок из файла:

reports/model_features.csv

Пример:

features = pd.read_csv("reports/model_features.csv")["feature"].tolist()
input_data = input_data[features]

Это снижает риск ошибки при интеграции.

14. Backend не должен подавать сырые таблицы напрямую

Модель не принимает напрямую исходные таблицы:

customers_live.csv
orders_live.csv
visits_live.csv
support_tickets_live.csv

Перед прогнозом необходимо сформировать агрегированный feature-vector.

Общая схема:

raw customer data
+ historical orders before snapshot_date
+ historical visits before snapshot_date
+ historical support tickets before snapshot_date
→ feature engineering
→ model_features
→ predict_proba
→ churn_probability
→ churn_prediction
15. Правило отсутствия data leakage

При расчёте признаков запрещено использовать события после даты прогнозирования.

Допустимо использовать только события:

event_date < snapshot_date

Для обучения целевая переменная строилась по будущему горизонту:

snapshot_date < delivered_order_date <= snapshot_date + 90 days

Для inference целевая переменная неизвестна и не рассчитывается.

16. Что такое snapshot_date

snapshot_date — дата, на которую строится прогноз.

Пример:

2023-12-01

Для этой даты признаки считаются только по событиям до 2023-12-01.

17. Ожидаемый backend API response

Рекомендуемый формат ответа API:

{
  "customer_id": 123,
  "snapshot_date": "2023-12-01",
  "churn_probability": 0.318742,
  "churn_prediction": 1,
  "threshold": 0.25,
  "model_name": "CatBoostClassifier"
}
18. Возможные ошибки при интеграции
18.1. Не хватает признаков

Ошибка:

Missing features: {...}

Причина: backend не сформировал один или несколько признаков из reports/model_features.csv.

Решение: добавить расчёт отсутствующих признаков в feature engineering.

18.2. Лишние признаки

Если во входном DataFrame есть лишние поля, их нужно удалить перед прогнозом.

Пример:

input_data = input_data[features]
18.3. Неверный тип категориального признака

Категориальные признаки должны быть строками:

city
gender
preferred_payment

Решение:

for col in ["city", "gender", "preferred_payment"]:
    input_data[col] = input_data[col].astype(str)
18.4. Пропуски в числовых признаках

Если числовой признак отсутствует или равен NaN, рекомендуется заполнить его нулём:

input_data = input_data.fillna(0)

Для категориальных признаков лучше использовать:

input_data[["city", "gender", "preferred_payment"]] = (
    input_data[["city", "gender", "preferred_payment"]].fillna("unknown")
)
19. Минимальный порядок inference

Backend должен выполнять следующие шаги:

1. Получить customer_id и snapshot_date.
2. Собрать исторические события клиента до snapshot_date.
3. Рассчитать 36 признаков.
4. Проверить наличие всех признаков из reports/model_features.csv.
5. Привести порядок колонок к reports/model_features.csv.
6. Загрузить models/catboost_churn_model.pkl.
7. Получить churn_probability через predict_proba.
8. Применить threshold = 0.25.
9. Вернуть churn_probability и churn_prediction.
20. Зависимости

Основные Python-зависимости указаны в:

requirements.txt

Для запуска:

python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
21. Контрольные артефакты проекта

После полного запуска пайплайна должны существовать:

data/ml_dataset.csv

models/catboost_churn_model.pkl
models/random_forest_baseline.pkl

reports/model_metrics.csv
reports/final_model_summary.csv
reports/catboost_threshold_selection.csv
reports/catboost_feature_importance.csv
reports/catboost_feature_importance.png
reports/model_features.csv
22. Итоговое решение

Для backend-интеграции использовать:

Model: models/catboost_churn_model.pkl
Threshold: 0.25
Input: 36 признаков из reports/model_features.csv
Output: churn_probability, churn_prediction