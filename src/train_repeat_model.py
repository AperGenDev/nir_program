
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

INPUT = Path("data/processed/repeat_prediction")
OUTPUT = INPUT / "models"

OUTPUT.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
NEGATIVE_RATIO = 2

TARGET = "target"

NUMERIC_FEATURES = [
    "user_orders_count",
    "user_total_items",
    "user_avg_cart_position",
    "user_avg_basket_size",
    "user_product_orders_count",
    "user_product_last_order_number",
    "user_product_recency",
    "user_product_reorder_rate",
    "user_product_avg_cart_position",
    "product_prior_orders",
    "product_prior_users",
    "product_reorder_rate",
    "order_dow",
    "order_hour_of_day",
    "days_since_prior_order",
]

CATEGORICAL_FEATURES = [
    "aisle_id",
    "department_id",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

print("1. Загрузка выборок...")

train = pd.read_parquet(INPUT / "train_repeat.parquet")
validation = pd.read_parquet(INPUT / "validation_repeat.parquet")
test = pd.read_parquet(INPUT / "test_repeat.parquet")

print(f"Train:      {train.shape}")
print(f"Validation: {validation.shape}")
print(f"Test:       {test.shape}")


# --------------------------------------------------
# 2. SAMPLE NEGATIVE EXAMPLES
# --------------------------------------------------

print("\n2. Формирование обучающей выборки...")

positive = train[train[TARGET] == 1]
negative = train[train[TARGET] == 0]

negative_sample = negative.sample(
    n=min(
        len(negative),
        len(positive) * NEGATIVE_RATIO
    ),
    random_state=RANDOM_STATE
)

train_sample = pd.concat(
    [positive, negative_sample],
    ignore_index=True
)

train_sample = train_sample.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)

print(f"Положительные примеры: {len(positive):,}")
print(f"Отрицательные примеры: {len(negative_sample):,}")
print(f"Всего для обучения: {len(train_sample):,}")


# --------------------------------------------------
# 3. PREPARE FEATURES
# --------------------------------------------------

print("\n3. Подготовка признаков...")

X_train = train_sample[FEATURES].copy()
y_train = train_sample[TARGET].astype(int)

X_validation = validation[FEATURES].copy()
y_validation = validation[TARGET].astype(int)

X_test = test[FEATURES].copy()
y_test = test[TARGET].astype(int)

# Categorical columns
for column in CATEGORICAL_FEATURES:
    for df in [X_train, X_validation, X_test]:
        df[column] = df[column].astype("Int64").astype(str)
        df[column] = df[column].replace("<NA>", "missing")


# --------------------------------------------------
# 4. LOGISTIC REGRESSION
# --------------------------------------------------

print("\n4. Обучение Logistic Regression...")

try:
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=True
    )
except TypeError:
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse=True
    )

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]),
            NUMERIC_FEATURES
        ),
        (
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", encoder)
            ]),
            CATEGORICAL_FEATURES
        )
    ]
)

logistic_model = Pipeline([
    ("preprocessor", preprocessor),
    (
        "classifier",
        LogisticRegression(
            max_iter=300,
            solver="saga",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    )
])

logistic_model.fit(X_train, y_train)

print("Logistic Regression обучена.")


# --------------------------------------------------
# 5. HIST GRADIENT BOOSTING
# --------------------------------------------------

print("\n5. Обучение HistGradientBoosting...")

# Для HGB используем нативные категориальные признаки.
hgb_train = X_train.copy()
hgb_validation = X_validation.copy()
hgb_test = X_test.copy()

for column in CATEGORICAL_FEATURES:
    categories = sorted(hgb_train[column].unique())

    hgb_train[column] = pd.Categorical(
        hgb_train[column],
        categories=categories
    )

    hgb_validation[column] = pd.Categorical(
        hgb_validation[column],
        categories=categories
    )

    hgb_test[column] = pd.Categorical(
        hgb_test[column],
        categories=categories
    )

try:
    hgb_model = HistGradientBoostingClassifier(
        max_iter=120,
        learning_rate=0.08,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        categorical_features="from_dtype",
        early_stopping=True,
        validation_fraction=0.1,
        random_state=RANDOM_STATE
    )

    hgb_model.fit(hgb_train, y_train)

except TypeError as error:
    raise RuntimeError(
        "Для HistGradientBoosting с категориальными "
        "признаками требуется scikit-learn версии 1.4 или новее. "
        "Обнови библиотеку: pip install -U scikit-learn"
    ) from error

print("HistGradientBoosting обучена.")


# --------------------------------------------------
# 6. EVALUATION
# --------------------------------------------------

print("\n6. Оценка моделей на validation-выборке...")

results = []


def evaluate_model(name, y_true, probabilities):
    predictions = (probabilities >= 0.5).astype(int)

    result = {
        "model": name,
        "roc_auc": roc_auc_score(y_true, probabilities),
        "pr_auc": average_precision_score(y_true, probabilities),
        "precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_true, predictions, zero_division=0
        ),
    }

    results.append(result)

    print(f"\n{name}")
    for key, value in result.items():
        if key != "model":
            print(f"{key}: {value:.6f}")

    return result


print("\nПредсказания Logistic Regression...")
logistic_probabilities = logistic_model.predict_proba(
    X_validation
)[:, 1]

evaluate_model(
    "LogisticRegression",
    y_validation,
    logistic_probabilities
)


print("\nПредсказания HistGradientBoosting...")
hgb_probabilities = hgb_model.predict_proba(
    hgb_validation
)[:, 1]

evaluate_model(
    "HistGradientBoosting",
    y_validation,
    hgb_probabilities
)


# --------------------------------------------------
# 7. SAVE MODELS AND RESULTS
# --------------------------------------------------

print("\n7. Сохранение моделей...")

joblib.dump(
    logistic_model,
    OUTPUT / "logistic_regression.joblib"
)

joblib.dump(
    hgb_model,
    OUTPUT / "hist_gradient_boosting.joblib"
)

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT / "model_comparison_validation.csv",
    index=False
)

print("\n=== СРАВНЕНИЕ МОДЕЛЕЙ ===")
print(results_df.to_string(index=False))

print(f"\nМодели и результаты сохранены в: {OUTPUT}")
