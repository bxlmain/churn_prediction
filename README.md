# Churn Prediction ML Pipeline

Проект реализует ML-пайплайн прогнозирования оттока клиентов интернет-магазина на основе исторических данных о клиентах, заказах, посещениях сайта и обращениях в поддержку.

Модель предназначена для выявления клиентов, склонных к оттоку, с целью последующего использования в сервисе удержания клиентов.

---

## 1. Цель проекта

Цель проекта — разработать модель машинного обучения, которая прогнозирует вероятность оттока клиента интернет-магазина.

В рамках проекта отток определяется следующим образом:

```text
churn_flag = 1
```

если клиент не совершил ни одного успешного заказа со статусом `delivered` в течение 90 дней после даты прогнозирования `snapshot_date`.

```text
churn_flag = 0
```

если клиент совершил хотя бы один успешный заказ со статусом `delivered` в течение 90 дней после `snapshot_date`.

---

## 2. Используемые данные

В проекте используются четыре исходные таблицы:

```text
data/customers_live.csv
data/orders_live.csv
data/visits_live.csv
data/support_tickets_live.csv
```

### 2.1. customers_live.csv

Таблица содержит информацию о клиентах:

- `customer_id`
- `registration_date`
- `city`
- `age`
- `gender`
- `preferred_payment`
- и другие клиентские атрибуты

### 2.2. orders_live.csv

Таблица содержит историю заказов:

- `order_id`
- `customer_id`
- `order_date`
- `amount`
- `quantity`
- `status`
- и другие параметры заказа

Ключевое поле для построения целевой переменной:

```text
status = delivered
```

### 2.3. visits_live.csv

Таблица содержит данные о посещениях сайта:

- `visit_id`
- `customer_id`
- `visit_time`
- `pages_viewed`
- `cart_adds`
- `checkout_initiated`
- `session_duration_sec`

### 2.4. support_tickets_live.csv

Таблица содержит данные об обращениях в поддержку:

- `ticket_id`
- `customer_id`
- `created_date`
- `closed_date`
- `rating`

---

## 3. Методология построения ML-датасета

Итоговый датасет строится в формате:

```text
одна строка = один клиент на одну дату прогнозирования
```

То есть один и тот же клиент может присутствовать в датасете несколько раз, если для него существуют разные `snapshot_date`.

Используемые даты прогнозирования:

```text
2023-04-01
2023-05-01
2023-06-01
2023-07-01
2023-08-01
2023-09-01
2023-10-01
2023-11-01
2023-12-01
```

Итоговый файл с признаками сохраняется в:

```text
data/ml_dataset.csv
```

---

## 4. Исключение data leakage

При формировании признаков используется строгое временное правило:

```text
event_date < snapshot_date
```

То есть признаки рассчитываются только по событиям, произошедшим до даты прогнозирования.

Целевая переменная рассчитывается по будущему горизонту:

```text
snapshot_date < delivered_order_date <= snapshot_date + 90 days
```

Таким образом, будущие события не попадают в признаки модели.

---

## 5. Признаки модели

В модели используются клиентские, транзакционные, поведенческие и сервисные признаки.

Основные группы признаков:

- клиентские характеристики;
- длительность жизненного цикла клиента;
- история заказов;
- сумма и средний чек заказов;
- количество доставленных и отменённых заказов;
- активность за последние 30 и 90 дней;
- посещения сайта;
- действия с корзиной;
- обращения в поддержку.

Полный список признаков модели сохраняется в:

```text
reports/model_features.csv
```

Количество входных признаков:

```text
36
```

Категориальные признаки:

```text
city
gender
preferred_payment
```

---

## 6. Разделение данных

Разделение выполнено по времени:

```text
Train:
2023-04-01 — 2023-09-01

Validation:
2023-10-01 — 2023-11-01

OOT:
2023-12-01
```

Такой подход имитирует реальную эксплуатацию модели, когда обучение выполняется на исторических данных, а проверка — на будущих периодах.

Размеры выборок:

```text
Train:      3987 строк
Validation: 1608 строк
OOT:        858 строк
```

---

## 7. Используемые модели

В проекте реализованы две модели:

### 7.1. Baseline-модель

```text
RandomForestClassifier
```

Baseline используется для сравнения качества основной модели.

### 7.2. Основная модель

```text
CatBoostClassifier
```

CatBoost выбран как основная модель, поскольку он хорошо работает с табличными данными и позволяет использовать категориальные признаки без ручного One-Hot Encoding.

---

## 8. Подбор threshold

После обучения CatBoost был выполнен подбор порога классификации на validation-выборке.

Финальный выбранный порог:

```text
threshold = 0.25
```

Правило классификации:

```text
если churn_probability >= 0.25 → churn_prediction = 1
если churn_probability < 0.25  → churn_prediction = 0
```

Порог `0.25` выбран по максимальному F1-score на validation-выборке и проверен на OOT-выборке.

---

## 9. Основные результаты

### 9.1. RandomForest на OOT

```text
ROC-AUC:   0.7601
PR-AUC:    0.6854
Precision: 0.7174
Recall:    0.4867
F1-score:  0.5800
```

### 9.2. CatBoost на OOT при threshold = 0.50

```text
ROC-AUC:   0.7371
PR-AUC:    0.6589
Precision: 0.7189
Recall:    0.3923
F1-score:  0.5076
```

### 9.3. CatBoost на OOT при threshold = 0.25

```text
ROC-AUC:   0.7371
PR-AUC:    0.6589
Precision: 0.5673
Recall:    0.6342
F1-score:  0.5989
```

Итоговая конфигурация:

```text
Model: CatBoostClassifier
Threshold: 0.25
```

После подбора порога модель стала выявлять больше клиентов, склонных к оттоку, что соответствует прикладной задаче удержания клиентов.

---

## 10. Интерпретация модели

Важность признаков CatBoost сохраняется в:

```text
reports/catboost_feature_importance.csv
reports/catboost_feature_importance.png
```

Наиболее значимые признаки:

- `age`
- `orders_amount_total`
- `customer_lifetime_days`
- `orders_amount_mean`
- `delivered_amount_mean`
- `orders_quantity_total`
- `delivered_amount_total`
- `orders_total`
- `city`
- `days_since_last_order`

Полученные результаты показывают, что модель использует как клиентские характеристики, так и признаки покупательской активности, давности последнего заказа и взаимодействия с сайтом.

---

## 11. Структура проекта

```text
churn_prediction/
│
├── configs/
│   └── config.yaml
│
├── data/
│   ├── customers_live.csv
│   ├── orders_live.csv
│   ├── visits_live.csv
│   ├── support_tickets_live.csv
│   └── ml_dataset.csv
│
├── models/
│   ├── random_forest_baseline.pkl
│   └── catboost_churn_model.pkl
│
├── reports/
│   ├── model_metrics.csv
│   ├── final_model_summary.csv
│   ├── catboost_threshold_selection.csv
│   ├── catboost_feature_importance.csv
│   ├── catboost_feature_importance.png
│   └── model_features.csv
│
├── src/
│   ├── data/
│   │   └── load_data.py
│   ├── features/
│   │   └── build_features.py
│   ├── models/
│   │   ├── train_model.py
│   │   └── predict.py
│   ├── evaluation/
│   │   ├── feature_importance.py
│   │   ├── final_summary.py
│   │   └── metrics.py
│   └── utils/
│       └── helpers.py
│
├── BACKEND_HANDOFF.md
├── MODEL_CONTRACT.md
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 12. Установка и запуск

### 12.1. Создание виртуального окружения

```powershell
python -m venv venv
```

### 12.2. Активация окружения

```powershell
.\venv\Scripts\Activate.ps1
```

Если PowerShell запрещает активацию окружения, выполнить:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Затем повторить активацию:

```powershell
.\venv\Scripts\Activate.ps1
```

### 12.3. Установка зависимостей

```powershell
pip install -r requirements.txt
```

### 12.4. Запуск полного пайплайна

```powershell
python main.py
```

После запуска выполняются следующие шаги:

```text
1. Загрузка исходных CSV-файлов
2. Формирование snapshot dataset
3. Построение признаков
4. Обучение RandomForest baseline
5. Обучение CatBoostClassifier
6. Подбор threshold
7. Оценка качества на Validation и OOT
8. Сохранение моделей и отчётов
```

---

## 13. Основные артефакты

После успешного запуска создаются следующие файлы:

```text
data/ml_dataset.csv

models/random_forest_baseline.pkl
models/catboost_churn_model.pkl

reports/model_metrics.csv
reports/final_model_summary.csv
reports/catboost_threshold_selection.csv
reports/catboost_feature_importance.csv
reports/catboost_feature_importance.png
reports/model_features.csv
```

---

## 14. Использование модели для прогноза

Для прогноза используется файл:

```text
src/models/predict.py
```

Основная функция:

```python
predict_churn(input_data: pd.DataFrame) -> pd.DataFrame
```

Функция возвращает:

```text
churn_probability
churn_prediction
```

Где:

```text
churn_probability — вероятность оттока клиента
churn_prediction — бинарный прогноз:
    1 = клиент склонен к оттоку
    0 = клиент не склонен к оттоку
```

---

## 15. Backend-интеграция

Для backend-разработчика подготовлены документы:

```text
MODEL_CONTRACT.md
BACKEND_HANDOFF.md
```

`MODEL_CONTRACT.md` описывает контракт модели: входные признаки, формат данных, threshold и выходной формат.

`BACKEND_HANDOFF.md` описывает практические шаги интеграции модели в backend-сервис.

Backend не должен подавать в модель сырые таблицы напрямую. Перед прогнозированием необходимо сформировать агрегированный feature-vector из 36 признаков в формате `reports/model_features.csv`.

---

## 16. Git

Рекомендуемый минимальный набор файлов для передачи:

```text
src/
configs/
data/
models/
reports/
main.py
requirements.txt
README.md
MODEL_CONTRACT.md
BACKEND_HANDOFF.md
.gitignore
```

Виртуальное окружение `venv/` не загружается в Git.

---

## 17. Ограничения проекта

1. Используемые данные являются синтетическими.
2. Высокая значимость демографических признаков требует осторожной интерпретации.
3. В реальной эксплуатации необходим мониторинг качества модели во времени.
4. При изменении структуры данных требуется повторная проверка feature engineering.
5. Для production-сценария рекомендуется вынести хранение модели и данных в отдельное хранилище.

---

## 18. Итог

В проекте реализован полный ML-пайплайн прогнозирования оттока клиентов:

```text
raw data
→ snapshot dataset
→ feature engineering
→ time-based split
→ baseline model
→ CatBoost model
→ threshold tuning
→ OOT evaluation
→ model artifacts
```

Финальная модель:

```text
CatBoostClassifier
```

Финальный threshold:

```text
0.25
```