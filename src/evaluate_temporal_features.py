
from pathlib import Path

import pandas as pd

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
OUTPUT = INPUT / "experiments"

OUTPUT.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
NEGATIVE_RATIO = 2
THRESHOLD = 0.5

NUMERIC_BASE = [
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
]

TEMPORAL = [
    "order_dow",
    "order_hour_of_day",
    "days_since_prior_order",
]

CATEGORICAL = [
    "aisle_id",
    "department_id",
]

FEATURE_SETS = {
    "A_behavioral": NUMERIC_BASE + CATEGORICAL,
    "B_behavioral_temporal": (
        NUMERIC_BASE + TEMPORAL + CATEGORICAL
    ),
}


# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

print("1. Загрузка данных...")

train = pd.read_parquet(INPUT / "train_repeat.parquet")
test = pd.read_parquet(INPUT / "test_repeat.parquet")

print("Train:", train.shape)
print("Test:", test.shape)


# --------------------------------------------------
# 2. SAME TRAINING SAMPLE
# --------------------------------------------------

print("\n2. Формирование обучающей выборки...")

positive = train[train["target"] == 1]
negative = train[train["target"] == 0]

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

print("Размер обучающей выборки:", len(train_sample))


# --------------------------------------------------
# 3. EVALUATION FUNCTION
# --------------------------------------------------

results = []

def prepare_features(data, features):
    X = data[features].copy()

    for column in CATEGORICAL:
        if column in features:
            X[column] = (
                X[column]
                .astype("Int64")
                .astype(str)
                .replace("<NA>", "missing")
            )

    return X


def evaluate(name, features):
    print(f"\nОбучение модели {name}...")

    X_train = prepare_features(train_sample, features)
    X_test = prepare_features(test, features)

    y_train = train_sample["target"].astype(int)
    y_test = test["target"].astype(int)

    # Категориальные признаки
    for column in CATEGORICAL:
        if column in features:
            categories = sorted(
                set(X_train[column].unique())
                | set(X_test[column].unique())
            )

            X_train[column] = pd.Categorical(
                X_train[column],
                categories=categories
            )

            X_test[column] = pd.Categorical(
                X_test[column],
                categories=categories
            )

    model = HistGradientBoostingClassifier(
        max_iter=120,
        learning_rate=0.08,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        categorical_features="from_dtype",
        early_stopping=True,
        validation_fraction=0.1,
        random_state=RANDOM_STATE
    )

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= THRESHOLD).astype(int)

    metrics = {
        "experiment": name,
        "features_count": len(features),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "pr_auc": average_precision_score(y_test, probabilities),
        "precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_test, predictions, zero_division=0
        ),
    }

    results.append(metrics)

    print(pd.Series(metrics).to_string())

    return model


# --------------------------------------------------
# 4. RUN EXPERIMENTS
# --------------------------------------------------

models = {}

for name, features in FEATURE_SETS.items():
    models[name] = evaluate(name, features)


# --------------------------------------------------
# 5. SAVE RESULTS
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT / "temporal_feature_comparison.csv",
    index=False
)

print("\n=== СРАВНЕНИЕ ВРЕМЕННЫХ ПРИЗНАКОВ ===")
print(results_df.to_string(index=False))

print("\n=== ИЗМЕНЕНИЕ МЕТРИК ===")

if len(results_df) == 2:
    baseline = results_df.iloc[0]
    temporal = results_df.iloc[1]

    for metric in [
        "roc_auc",
        "pr_auc",
        "precision",
        "recall",
        "f1"
    ]:
        difference = temporal[metric] - baseline[metric]
        print(f"{metric}: {difference:+.6f}")

print(f"\nРезультаты сохранены в {OUTPUT}")
