from pathlib import Path
import math

import duckdb
import joblib
import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw"
REPEAT_DIR = ROOT / "data" / "processed" / "repeat_prediction"

MODEL_FILE = REPEAT_DIR / "models" / "hist_gradient_boosting.joblib"
TRAIN_FILE = REPEAT_DIR / "train_repeat.parquet"
VALIDATION_FILE = REPEAT_DIR / "validation_repeat.parquet"
TRAIN_PRODUCTS_FILE = RAW_DIR / "order_products__train.csv"

GRAPH_DIR = (
    ROOT
    / "data"
    / "processed"
    / "graph_experiments"
    / "top_2000_pair_100_lift_1.2"
)

OUTPUT_DIR = ROOT / "data" / "processed" / "cascade_evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

K_VALUES = [5, 10, 20]
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]

# Должны совпадать с настройками train_repeat_model.py.
RANDOM_STATE = 42
NEGATIVE_RATIO = 2
TARGET = "target"

CATEGORICAL_FEATURES = [
    "aisle_id",
    "department_id",
]


# ============================================================
# METRICS
# ============================================================

def precision_at_k(recommended, relevant, k):
    hits = sum(product_id in relevant for product_id in recommended[:k])
    return hits / k


def recall_at_k(recommended, relevant, k):
    if not relevant:
        return 0.0

    hits = sum(product_id in relevant for product_id in recommended[:k])
    return hits / len(relevant)


def average_precision_at_k(recommended, relevant, k):
    if not relevant:
        return 0.0

    score = 0.0
    hits = 0

    for rank, product_id in enumerate(recommended[:k], start=1):
        if product_id in relevant:
            hits += 1
            score += hits / rank

    return score / min(len(relevant), k)


def ndcg_at_k(recommended, relevant, k):
    if not relevant:
        return 0.0

    dcg = 0.0

    for rank, product_id in enumerate(recommended[:k], start=1):
        if product_id in relevant:
            dcg += 1.0 / math.log2(rank + 1)

    ideal_hits = min(len(relevant), k)
    idcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_hits + 1)
    )

    return dcg / idcg if idcg else 0.0


# ============================================================
# SCORE NORMALIZATION
# ============================================================

def normalize_scores(scores):
    """Min-max нормализация оценок внутри одного пользователя."""
    if not scores:
        return {}

    values = np.array(list(scores.values()), dtype=float)
    min_value = values.min()
    max_value = values.max()

    if max_value == min_value:
        return {product_id: 1.0 for product_id in scores}

    return {
        product_id: (score - min_value) / (max_value - min_value)
        for product_id, score in scores.items()
    }


# ============================================================
# 1. LOAD VALIDATION DATA
# ============================================================

print("=" * 70)
print("1. ЗАГРУЗКА VALIDATION-ДАННЫХ")
print("=" * 70)

if not VALIDATION_FILE.exists():
    raise FileNotFoundError(f"Не найден файл: {VALIDATION_FILE}")

validation_df = pd.read_parquet(VALIDATION_FILE)

required_columns = {
    "user_id", "product_id", "target_order_id"
}
missing = required_columns - set(validation_df.columns)
if missing:
    raise ValueError(
        f"В validation-файле отсутствуют обязательные колонки: {sorted(missing)}"
    )

validation_df["user_id"] = validation_df["user_id"].astype(int)
validation_df["product_id"] = validation_df["product_id"].astype(int)

validation_users = sorted(validation_df["user_id"].unique().tolist())

print(f"Строк validation-выборки: {len(validation_df):,}")
print(f"Validation-пользователей: {len(validation_users):,}")

user_history = (
    validation_df
    .groupby("user_id")["product_id"]
    .apply(lambda values: set(values.astype(int)))
    .to_dict()
)


# ============================================================
# 2. LOAD VALIDATION TARGET BASKETS
# ============================================================

print("\n" + "=" * 70)
print("2. ЗАГРУЗКА ЦЕЛЕВЫХ ПОКУПОК")
print("=" * 70)

if not TRAIN_PRODUCTS_FILE.exists():
    raise FileNotFoundError(f"Не найден файл: {TRAIN_PRODUCTS_FILE}")

orders_df = (
    validation_df[["user_id", "target_order_id"]]
    .drop_duplicates()
)

if orders_df["user_id"].duplicated().any():
    raise ValueError(
        "Для пользователя обнаружено несколько target_order_id."
    )

con = duckdb.connect()
try:
    con.register("validation_orders_df", orders_df)
    target_path = TRAIN_PRODUCTS_FILE.resolve().as_posix().replace("'", "''")

    target_df = con.execute(f"""
        SELECT
            t.user_id,
            p.product_id
        FROM validation_orders_df t
        JOIN read_csv_auto(
            '{target_path}',
            header = true
        ) p
            ON t.target_order_id = p.order_id
        GROUP BY
            t.user_id,
            p.product_id
    """).df()
finally:
    con.close()

if target_df.empty:
    raise ValueError(
        "Не удалось загрузить целевые корзины validation-пользователей. "
        "Проверь target_order_id и order_products__train.csv."
    )

target_df["user_id"] = target_df["user_id"].astype(int)
target_df["product_id"] = target_df["product_id"].astype(int)

user_targets = (
    target_df
    .groupby("user_id")["product_id"]
    .apply(lambda values: set(values.astype(int)))
    .to_dict()
)

# Оставляем только пользователей, для которых найдена целевая корзина.
validation_users = [
    user_id for user_id in validation_users
    if user_id in user_targets
]

if not validation_users:
    raise ValueError("Не найдено ни одного пользователя с целевой корзиной.")

user_history = {
    user_id: user_history[user_id]
    for user_id in validation_users
}

all_target_items = set(target_df["product_id"].astype(int).unique())

print(f"Пользователей с target: {len(user_targets):,}")
print(f"Пользователей в оценке: {len(validation_users):,}")
print(f"Уникальных целевых товаров: {len(all_target_items):,}")


# ============================================================
# 3. LOAD MODEL AND RECREATE TRAINING CATEGORIES
# ============================================================

print("\n" + "=" * 70)
print("3. ЗАГРУЗКА МОДЕЛИ ПОВТОРНЫХ ПОКУПОК")
print("=" * 70)

if not MODEL_FILE.exists():
    raise FileNotFoundError(f"Не найдена модель: {MODEL_FILE}")

if not TRAIN_FILE.exists():
    raise FileNotFoundError(f"Не найден train-файл: {TRAIN_FILE}")

model = joblib.load(MODEL_FILE)

if not hasattr(model, "predict_proba"):
    raise TypeError("Модель не поддерживает predict_proba().")

if not hasattr(model, "feature_names_in_"):
    raise ValueError(
        "У модели отсутствует feature_names_in_. "
        "Убедись, что HistGradientBoosting обучалась на DataFrame."
    )

feature_columns = list(model.feature_names_in_)
missing_columns = [
    column for column in feature_columns
    if column not in validation_df.columns
]

if missing_columns:
    raise ValueError(
        f"В validation-наборе отсутствуют признаки: {missing_columns}"
    )

# Восстанавливаем тот же train_sample и категории, которые использовались
# при обучении HGB. Это необходимо, потому что модель обучалась на pandas
# Categorical с категориями, сформированными по обучающим данным.
train_df = pd.read_parquet(TRAIN_FILE)

if TARGET not in train_df.columns:
    raise ValueError(f"В train-файле отсутствует целевая колонка {TARGET!r}.")

positive = train_df[train_df[TARGET] == 1]
negative = train_df[train_df[TARGET] == 0]

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

missing_cat_columns = [
    column for column in CATEGORICAL_FEATURES
    if column not in train_sample.columns
]
if missing_cat_columns:
    raise ValueError(
        f"В train-файле отсутствуют категориальные признаки: "
        f"{missing_cat_columns}"
    )

category_values = {}

for column in CATEGORICAL_FEATURES:
    values = (
        train_sample[column]
        .astype("Int64")
        .astype(str)
        .replace("<NA>", "missing")
    )
    category_values[column] = sorted(values.unique().tolist())

X_validation = validation_df[feature_columns].copy()

for column in CATEGORICAL_FEATURES:
    if column not in X_validation.columns:
        continue

    values = (
        X_validation[column]
        .astype("Int64")
        .astype(str)
        .replace("<NA>", "missing")
    )

    # Значения, которых не было в train, превращаются в NaN.
    X_validation[column] = pd.Categorical(
        values,
        categories=category_values[column]
    )

print(f"Количество признаков: {len(feature_columns)}")
print("Признаки:", feature_columns)
print("Категориальные признаки:", CATEGORICAL_FEATURES)
print("\nТипы подготовленных признаков:")
print(X_validation.dtypes)


# ============================================================
# 4. PREDICT REPEAT-PURCHASE SCORES
# ============================================================

print("\n" + "=" * 70)
print("4. РАСЧЕТ ВЕРОЯТНОСТЕЙ ПОВТОРНЫХ ПОКУПОК")
print("=" * 70)

try:
    repeat_probabilities = model.predict_proba(X_validation)[:, 1]
except Exception as error:
    raise RuntimeError(
        "Не удалось получить предсказания HistGradientBoosting. "
        "Проверь, что категории в этом скрипте сформированы точно так же, "
        "как в train_repeat_model.py."
    ) from error

print(f"Количество оценок: {len(repeat_probabilities):,}")
print(f"Минимальная оценка: {repeat_probabilities.min():.6f}")
print(f"Максимальная оценка: {repeat_probabilities.max():.6f}")

validation_df = validation_df.copy()
validation_df["repeat_score"] = repeat_probabilities

repeat_scores = {}

for row in validation_df[
    ["user_id", "product_id", "repeat_score"]
].itertuples(index=False):
    user_id = int(row.user_id)
    product_id = int(row.product_id)
    score = float(row.repeat_score)

    repeat_scores.setdefault(user_id, {})[product_id] = score

print("Оценки повторных покупок рассчитаны.")


# ============================================================
# 5. LOAD PRODUCT GRAPH
# ============================================================

print("\n" + "=" * 70)
print("5. ЗАГРУЗКА ГРАФА АССОЦИАЦИЙ")
print("=" * 70)

nodes_file = GRAPH_DIR / "graph_nodes.csv"
edges_file = GRAPH_DIR / "graph_edges.csv"

if not nodes_file.exists() or not edges_file.exists():
    raise FileNotFoundError(
        f"Не найдены graph_nodes.csv или graph_edges.csv в {GRAPH_DIR}"
    )

nodes_df = pd.read_csv(nodes_file)
edges_df = pd.read_csv(edges_file)

graph_products = set(nodes_df["product_id"].astype(int))

adjacency = {product_id: {} for product_id in graph_products}

for row in edges_df.itertuples(index=False):
    a = int(row.product_a)
    b = int(row.product_b)

    conf_ab = float(row.confidence_a_to_b)
    conf_ba = float(row.confidence_b_to_a)

    adjacency.setdefault(a, {})[b] = conf_ab
    adjacency.setdefault(b, {})[a] = conf_ba

print(f"Вершин: {len(graph_products):,}")
print(f"Рёбер: {len(edges_df):,}")


# ============================================================
# 6. GENERATE GRAPH SCORES
# ============================================================

print("\n" + "=" * 70)
print("6. ФОРМИРОВАНИЕ ГРАФОВЫХ РЕКОМЕНДАЦИЙ")
print("=" * 70)

graph_scores = {}

for index, user_id in enumerate(validation_users, start=1):
    history = user_history[user_id]
    scores = {}

    for product_id in history:
        neighbors = adjacency.get(product_id, {})

        for candidate_id, confidence in neighbors.items():
            if candidate_id in history:
                continue

            scores[candidate_id] = (
                scores.get(candidate_id, 0.0) + confidence
            )

    graph_scores[user_id] = scores

    if index % 5000 == 0:
        print(f"Обработано пользователей: {index:,}")

print("Графовые оценки рассчитаны.")


# ============================================================
# 7. NORMALIZE SCORES
# ============================================================

print("\n" + "=" * 70)
print("7. НОРМАЛИЗАЦИЯ ОЦЕНОК")
print("=" * 70)

repeat_scores_normalized = {}
graph_scores_normalized = {}

for user_id in validation_users:
    repeat_scores_normalized[user_id] = normalize_scores(
        repeat_scores.get(user_id, {})
    )
    graph_scores_normalized[user_id] = normalize_scores(
        graph_scores.get(user_id, {})
    )

print("Нормализация завершена.")


# ============================================================
# 8. BUILD CASCADE RECOMMENDATIONS
# ============================================================

def build_recommendations(user_id, alpha):
    repeat_user_scores = repeat_scores_normalized[user_id]
    graph_user_scores = graph_scores_normalized[user_id]

    repeat_candidates = set(repeat_user_scores)
    graph_candidates = set(graph_user_scores)

    if alpha == 1.0:
        candidates = repeat_candidates
    elif alpha == 0.0:
        candidates = graph_candidates
    else:
        candidates = repeat_candidates | graph_candidates

    combined_scores = {}

    for product_id in candidates:
        repeat_score = repeat_user_scores.get(product_id, 0.0)
        graph_score = graph_user_scores.get(product_id, 0.0)

        combined_scores[product_id] = (
            alpha * repeat_score
            + (1.0 - alpha) * graph_score
        )

    ranked = sorted(
        combined_scores.items(),
        key=lambda item: (-item[1], item[0])
    )

    return [product_id for product_id, score in ranked]


# ============================================================
# 9. EVALUATE ALL ALPHAS ON VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("8. ОЦЕНКА КАСКАДНОЙ МОДЕЛИ НА VALIDATION")
print("=" * 70)

all_metrics = []
all_coverage = []

for alpha in ALPHAS:
    print(f"\nОценка alpha = {alpha:.2f}")

    recommended_by_user = {
        user_id: build_recommendations(user_id, alpha)
        for user_id in validation_users
    }

    for k in K_VALUES:
        precisions = []
        recalls = []
        aps = []
        ndcgs = []

        unique_recommended = set()
        users_with_recommendations = 0
        users_with_k_recommendations = 0
        recommendation_counts = []

        for user_id in validation_users:
            recommended = recommended_by_user[user_id]
            relevant = user_targets[user_id]
            top_k = recommended[:k]

            unique_recommended.update(top_k)
            recommendation_counts.append(len(top_k))

            if top_k:
                users_with_recommendations += 1

            if len(top_k) >= k:
                users_with_k_recommendations += 1

            precisions.append(precision_at_k(recommended, relevant, k))
            recalls.append(recall_at_k(recommended, relevant, k))
            aps.append(average_precision_at_k(recommended, relevant, k))
            ndcgs.append(ndcg_at_k(recommended, relevant, k))

        metrics = {
            "alpha": alpha,
            "repeat_weight": alpha,
            "graph_weight": 1.0 - alpha,
            "K": k,
            "users_evaluated": len(validation_users),
            "precision": float(np.mean(precisions)),
            "recall": float(np.mean(recalls)),
            "map": float(np.mean(aps)),
            "ndcg": float(np.mean(ndcgs)),
            "unique_recommended_products": len(unique_recommended),
            "recommendation_catalog_coverage": (
                len(unique_recommended) / len(all_target_items)
                if all_target_items else 0.0
            ),
            "users_with_recommendations": users_with_recommendations,
            "users_with_k_recommendations": users_with_k_recommendations,
            "mean_recommendation_count": float(
                np.mean(recommendation_counts)
            ),
        }

        all_metrics.append(metrics)

        print(
            f"K={k:2d} | "
            f"Precision={metrics['precision']:.5f} | "
            f"Recall={metrics['recall']:.5f} | "
            f"MAP={metrics['map']:.5f} | "
            f"NDCG={metrics['ndcg']:.5f}"
        )

    all_recommended = set()
    for user_id in validation_users:
        all_recommended.update(recommended_by_user[user_id][:20])

    all_coverage.append({
        "alpha": alpha,
        "repeat_weight": alpha,
        "graph_weight": 1.0 - alpha,
        "validation_users": len(validation_users),
        "unique_recommended_products_at_20": len(all_recommended),
        "target_catalog_size": len(all_target_items),
    })


# ============================================================
# 10. EVALUATE BY USER HISTORY SIZE
# ============================================================

print("\n" + "=" * 70)
print("9. ДОПОЛНИТЕЛЬНЫЙ ЭКСПЕРИМЕНТ: РАЗМЕР ИСТОРИИ")
print("=" * 70)

HISTORY_GROUPS = {
    "A_short": lambda n: n < 5,
    "B_medium": lambda n: 5 <= n < 20,
    "C_long": lambda n: n >= 20,
}

GROUP_ALPHAS = [0.75, 1.0]
GROUP_K = 20

users_by_group = {
    group_name: []
    for group_name in HISTORY_GROUPS
}

for user_id in validation_users:
    history_size = len(user_history[user_id])

    for group_name, condition in HISTORY_GROUPS.items():
        if condition(history_size):
            users_by_group[group_name].append(user_id)
            break

print("\nРаспределение пользователей:")
for group_name, users in users_by_group.items():
    print(f"{group_name}: {len(users):,} пользователей")

group_metrics = []

for alpha in GROUP_ALPHAS:
    print(f"\nОценка alpha = {alpha:.2f}")

    recommendations = {
        user_id: build_recommendations(user_id, alpha)
        for user_id in validation_users
    }

    for group_name, group_users in users_by_group.items():
        if not group_users:
            continue

        precisions = []
        recalls = []
        aps = []
        ndcgs = []
        users_with_k_recommendations = 0
        recommendation_counts = []

        for user_id in group_users:
            recommended = recommendations[user_id]
            relevant = user_targets[user_id]
            top_k = recommended[:GROUP_K]

            recommendation_counts.append(len(top_k))

            if len(top_k) >= GROUP_K:
                users_with_k_recommendations += 1

            precisions.append(
                precision_at_k(recommended, relevant, GROUP_K)
            )
            recalls.append(
                recall_at_k(recommended, relevant, GROUP_K)
            )
            aps.append(
                average_precision_at_k(recommended, relevant, GROUP_K)
            )
            ndcgs.append(
                ndcg_at_k(recommended, relevant, GROUP_K)
            )

        mean_history_size = float(np.mean([
            len(user_history[user_id])
            for user_id in group_users
        ]))

        result = {
            "alpha": alpha,
            "history_group": group_name,
            "users_count": len(group_users),
            "mean_history_size": mean_history_size,
            "precision_at_20": float(np.mean(precisions)),
            "recall_at_20": float(np.mean(recalls)),
            "map_at_20": float(np.mean(aps)),
            "ndcg_at_20": float(np.mean(ndcgs)),
            "users_with_20_recommendations": users_with_k_recommendations,
            "share_with_20_recommendations": (
                users_with_k_recommendations / len(group_users)
            ),
            "mean_recommendation_count": float(
                np.mean(recommendation_counts)
            ),
        }

        group_metrics.append(result)

        print(
            f"{group_name:10s} | "
            f"Users={len(group_users):6,} | "
            f"P@20={result['precision_at_20']:.5f} | "
            f"R@20={result['recall_at_20']:.5f} | "
            f"MAP={result['map_at_20']:.5f} | "
            f"NDCG={result['ndcg_at_20']:.5f}"
        )

group_metrics_df = pd.DataFrame(group_metrics)
group_metrics_path = OUTPUT_DIR / "cascade_validation_history_group_metrics.csv"
group_metrics_df.to_csv(group_metrics_path, index=False)

print("\nРезультаты эксперимента по размеру истории:")
print(group_metrics_df.to_string(index=False))
print(f"Файл: {group_metrics_path}")


# ============================================================
# 11. SAVE VALIDATION RESULTS
# ============================================================

print("\n" + "=" * 70)
print("10. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
print("=" * 70)

metrics_df = pd.DataFrame(all_metrics)
coverage_df = pd.DataFrame(all_coverage)

metrics_path = OUTPUT_DIR / "cascade_validation_metrics.csv"
coverage_path = OUTPUT_DIR / "cascade_validation_coverage.csv"

metrics_df.to_csv(metrics_path, index=False)
coverage_df.to_csv(coverage_path, index=False)

summary_df = metrics_df.pivot(
    index=["alpha", "repeat_weight", "graph_weight"],
    columns="K",
    values=["precision", "recall", "map", "ndcg"]
)

summary_df.columns = [
    f"{metric}_at_{k}"
    for metric, k in summary_df.columns
]
summary_df = summary_df.reset_index()

summary_path = OUTPUT_DIR / "cascade_validation_summary.csv"
summary_df.to_csv(summary_path, index=False)

print("\nИТОГОВАЯ СВОДКА КАСКАДНОЙ МОДЕЛИ НА VALIDATION")
print(summary_df.to_string(index=False))

print("\nФайлы результатов:")
print(metrics_path)
print(coverage_path)
print(summary_path)
print(group_metrics_path)

print("\nЭксперимент завершен.")
print(
    "Важно: alpha выбирается по validation. "
    "Для финальной оценки используй test_repeat.parquet "
    "и только выбранное значение alpha."
)
