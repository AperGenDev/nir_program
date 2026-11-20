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
TEST_FILE = REPEAT_DIR / "test_repeat.parquet"
TRAIN_PRODUCTS_FILE = RAW_DIR / "order_products__train.csv"
ORDERS_FILE = RAW_DIR / "orders.csv"
PRIOR_FILE = RAW_DIR / "order_products__prior.csv"

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
TOP_K = 20

# Должны совпадать с настройками train_repeat_model.py.
RANDOM_STATE = 42
NEGATIVE_RATIO = 2
TARGET = "target"

CATEGORICAL_FEATURES = ["aisle_id", "department_id"]

# Разбиение пользователей должно совпадать с prepare_repeat_dataset.py.
# test: hash % 10 in (0, 5)
# test: hash % 10 == 1
# train: все остальные значения.


# ============================================================
# METRICS
# ============================================================

def precision_at_k(recommended, relevant, k):
    return sum(item in relevant for item in recommended[:k]) / k


def recall_at_k(recommended, relevant, k):
    if not relevant:
        return 0.0
    return sum(item in relevant for item in recommended[:k]) / len(relevant)


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

    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, product_id in enumerate(recommended[:k], start=1)
        if product_id in relevant
    )

    ideal_hits = min(len(relevant), k)
    idcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_hits + 1)
    )
    return dcg / idcg if idcg else 0.0


def sql_path(path):
    return path.resolve().as_posix().replace("'", "''")


# ============================================================
# 1. LOAD TEST DATA AND TARGET BASKETS
# ============================================================

print("=" * 70)
print("1. ЗАГРУЗКА TEST-ДАННЫХ")
print("=" * 70)

for path in [TEST_FILE, TRAIN_PRODUCTS_FILE, ORDERS_FILE, PRIOR_FILE,
             TRAIN_FILE, MODEL_FILE]:
    if not path.exists():
        raise FileNotFoundError(f"Не найден файл: {path}")

test_df = pd.read_parquet(TEST_FILE)

required_columns = {"user_id", "product_id", "target_order_id"}
missing = required_columns - set(test_df.columns)
if missing:
    raise ValueError(f"В test-файле отсутствуют колонки: {sorted(missing)}")

test_df["user_id"] = test_df["user_id"].astype(int)
test_df["product_id"] = test_df["product_id"].astype(int)

test_users = sorted(test_df["user_id"].unique().tolist())
print(f"Строк test-выборки: {len(test_df):,}")
print(f"Test-пользователей: {len(test_users):,}")

# В данном наборе product_id — товары-кандидаты модели повторных покупок,
# то есть товары из предыдущей истории пользователя.
user_history = (
    test_df.groupby("user_id")["product_id"]
    .apply(lambda values: set(values.astype(int)))
    .to_dict()
)

print("\nЗагрузка целевых корзин...")
orders_df = test_df[["user_id", "target_order_id"]].drop_duplicates()

if orders_df["user_id"].duplicated().any():
    raise ValueError("Для пользователя найдено несколько target_order_id.")

con = duckdb.connect()
try:
    con.register("test_orders_df", orders_df)
    target_df = con.execute(f"""
        SELECT
            t.user_id,
            p.product_id
        FROM test_orders_df t
        JOIN read_csv_auto('{sql_path(TRAIN_PRODUCTS_FILE)}') p
            ON t.target_order_id = p.order_id
        GROUP BY t.user_id, p.product_id
    """).df()
finally:
    con.close()

if target_df.empty:
    raise ValueError("Целевые корзины test не найдены.")

target_df["user_id"] = target_df["user_id"].astype(int)
target_df["product_id"] = target_df["product_id"].astype(int)

user_targets = (
    target_df.groupby("user_id")["product_id"]
    .apply(lambda values: set(values.astype(int)))
    .to_dict()
)

test_users = [u for u in test_users if u in user_targets]
user_history = {u: user_history[u] for u in test_users}

# Новые товары — товары целевого заказа, которых не было в предыдущей истории.
user_new_targets = {
    user_id: user_targets[user_id] - user_history[user_id]
    for user_id in test_users
}

print(f"Пользователей с target: {len(user_targets):,}")
print(f"Пользователей в оценке: {len(test_users):,}")
print(f"Уникальных целевых товаров: {target_df['product_id'].nunique():,}")
print(
    "Новых целевых товаров (пользователь-товар): "
    f"{sum(len(items) for items in user_new_targets.values()):,}"
)


# ============================================================
# 2. LOAD MODEL AND TRAIN-ONLY CATEGORY VOCABULARY
# ============================================================

print("\n" + "=" * 70)
print("2. ЗАГРУЗКА МОДЕЛИ ПОВТОРНЫХ ПОКУПОК")
print("=" * 70)

model = joblib.load(MODEL_FILE)
if not hasattr(model, "predict_proba"):
    raise TypeError("Модель не поддерживает predict_proba().")
if not hasattr(model, "feature_names_in_"):
    raise ValueError("У модели отсутствует feature_names_in_.")

feature_columns = list(model.feature_names_in_)
missing_columns = [c for c in feature_columns if c not in test_df.columns]
if missing_columns:
    raise ValueError(f"В test отсутствуют признаки: {missing_columns}")

# Точно воспроизводим train negative sampling, чтобы получить словарь
# категорий, сформированный исключительно по обучающим данным.
train_df = pd.read_parquet(TRAIN_FILE)
positive = train_df[train_df[TARGET] == 1]
negative = train_df[train_df[TARGET] == 0]

negative_sample = negative.sample(
    n=min(len(negative), len(positive) * NEGATIVE_RATIO),
    random_state=RANDOM_STATE,
)
train_sample = pd.concat([positive, negative_sample], ignore_index=True)
train_sample = train_sample.sample(
    frac=1, random_state=RANDOM_STATE
).reset_index(drop=True)

category_values = {}
for column in CATEGORICAL_FEATURES:
    values = (
        train_sample[column]
        .astype("Int64")
        .astype(str)
        .replace("<NA>", "missing")
    )
    category_values[column] = sorted(values.unique().tolist())

X_test = test_df[feature_columns].copy()
for column in CATEGORICAL_FEATURES:
    values = (
        X_test[column]
        .astype("Int64")
        .astype(str)
        .replace("<NA>", "missing")
    )
    X_test[column] = pd.Categorical(
        values,
        categories=category_values[column],
    )

print(f"Количество признаков: {len(feature_columns)}")
print("Категориальные признаки:", CATEGORICAL_FEATURES)

print("\nРасчёт оценок повторных покупок...")
repeat_probabilities = model.predict_proba(X_test)[:, 1]
test_df = test_df.copy()
test_df["repeat_score"] = repeat_probabilities

repeat_scores = {}
for row in test_df[
    ["user_id", "product_id", "repeat_score"]
].itertuples(index=False):
    repeat_scores.setdefault(int(row.user_id), {})[
        int(row.product_id)
    ] = float(row.repeat_score)

# Сортируем повторные товары по вероятности модели.
repeat_ranked = {
    user_id: [
        product_id
        for product_id, _ in sorted(
            repeat_scores.get(user_id, {}).items(),
            key=lambda item: (-item[1], item[0]),
        )
    ]
    for user_id in test_users
}


# ============================================================
# 3. LOAD TRAIN-ONLY PRODUCT GRAPH
# ============================================================

print("\n" + "=" * 70)
print("3. ЗАГРУЗКА ГРАФА ТОВАРНЫХ ВЗАИМОСВЯЗЕЙ")
print("=" * 70)

nodes_file = GRAPH_DIR / "graph_nodes.csv"
edges_file = GRAPH_DIR / "graph_edges.csv"
if not nodes_file.exists() or not edges_file.exists():
    raise FileNotFoundError(
        f"Не найдены graph_nodes.csv / graph_edges.csv в {GRAPH_DIR}. "
        "Сначала запусти build_train_graph.py после исправления train split."
    )

nodes_df = pd.read_csv(nodes_file)
edges_df = pd.read_csv(edges_file)

graph_products = set(nodes_df["product_id"].astype(int))
adjacency = {product_id: {} for product_id in graph_products}

for row in edges_df.itertuples(index=False):
    a = int(row.product_a)
    b = int(row.product_b)
    adjacency.setdefault(a, {})[b] = float(row.confidence_a_to_b)
    adjacency.setdefault(b, {})[a] = float(row.confidence_b_to_a)

print(f"Вершин: {len(graph_products):,}")
print(f"Рёбер: {len(edges_df):,}")

graph_ranked = {}
for index, user_id in enumerate(test_users, start=1):
    history = user_history[user_id]
    scores = {}

    for product_id in history:
        for candidate_id, confidence in adjacency.get(product_id, {}).items():
            if candidate_id in history:
                continue
            scores[candidate_id] = scores.get(candidate_id, 0.0) + confidence

    graph_ranked[user_id] = [
        product_id
        for product_id, _ in sorted(
            scores.items(),
            key=lambda item: (-item[1], item[0]),
        )
    ]

    if index % 5000 == 0:
        print(f"Обработано пользователей: {index:,}")

print("Графовые кандидаты сформированы.")


# ============================================================
# 4. BUILD TRAIN-ONLY POPULARITY RANKING
# ============================================================

print("\n" + "=" * 70)
print("4. РАСЧЁТ ПОПУЛЯРНОСТИ ТОВАРОВ НА TRAIN")
print("=" * 70)

# Используются только prior-транзакции пользователей train-группы.
# Разбиение идентично prepare_repeat_dataset.py и build_train_graph.py.
con = duckdb.connect()
try:
    popularity_df = con.execute(f"""
        WITH train_users AS (
            SELECT DISTINCT user_id
            FROM read_csv_auto('{sql_path(ORDERS_FILE)}')
            WHERE eval_set = 'train'
              AND hash(user_id) % 10 NOT IN (0, 1, 5)
        ),
        train_orders AS (
            SELECT o.order_id
            FROM read_csv_auto('{sql_path(ORDERS_FILE)}') o
            JOIN train_users u ON o.user_id = u.user_id
            WHERE o.eval_set = 'prior'
        )
        SELECT
            p.product_id,
            COUNT(DISTINCT p.order_id) AS train_item_orders
        FROM read_csv_auto('{sql_path(PRIOR_FILE)}') p
        JOIN train_orders t ON p.order_id = t.order_id
        GROUP BY p.product_id
        ORDER BY train_item_orders DESC, p.product_id
    """).df()
finally:
    con.close()

popularity_ranked_all = popularity_df["product_id"].astype(int).tolist()
print(f"Товаров в рейтинге популярности: {len(popularity_ranked_all):,}")


# ============================================================
# 5. BUILD TWO FILLING STRATEGIES
# ============================================================

def fill_with_candidates(primary, fallback, history, limit=TOP_K):
    """Сохраняет порядок primary и дополняет список fallback-кандидатами."""
    result = []
    used = set()

    # Повторные покупки намеренно могут присутствовать в истории:
    # их как раз и предсказывает repeat-модель.
    for product_id in primary:
        if product_id in used:
            continue
        result.append(product_id)
        used.add(product_id)
        if len(result) == limit:
            return result

    # Дополняющие товары должны быть новыми для пользователя.
    for product_id in fallback:
        if product_id in history or product_id in used:
            continue
        result.append(product_id)
        used.add(product_id)
        if len(result) == limit:
            break

    return result


recommendations = {
    "repeat_only": {},
    "repeat_plus_graph": {},
    "repeat_plus_popularity": {},
}

for user_id in test_users:
    history = user_history[user_id]
    primary = repeat_ranked[user_id]

    recommendations["repeat_only"][user_id] = fill_with_candidates(
        primary, [], history
    )
    recommendations["repeat_plus_graph"][user_id] = fill_with_candidates(
        primary, graph_ranked[user_id], history
    )
    recommendations["repeat_plus_popularity"][user_id] = fill_with_candidates(
        primary, popularity_ranked_all, history
    )


# ============================================================
# 6. EVALUATE OVERALL QUALITY AND NEW-ITEM DISCOVERY
# ============================================================

print("\n" + "=" * 70)
print("5. СРАВНЕНИЕ СПОСОБОВ ЗАПОЛНЕНИЯ ДО 20 РЕКОМЕНДАЦИЙ")
print("=" * 70)

method_labels = {
    "repeat_only": "Повторные покупки без дополнения",
    "repeat_plus_graph": "Повторные покупки + граф",
    "repeat_plus_popularity": "Повторные покупки + популярность",
}

metric_rows = []
new_item_rows = []

for method, per_user_recs in recommendations.items():
    print(f"\n{method_labels[method]}")

    for k in K_VALUES:
        precision_values = []
        recall_values = []
        ap_values = []
        ndcg_values = []

        for user_id in test_users:
            recs = per_user_recs[user_id]
            targets = user_targets[user_id]

            precision_values.append(precision_at_k(recs, targets, k))
            recall_values.append(recall_at_k(recs, targets, k))
            ap_values.append(average_precision_at_k(recs, targets, k))
            ndcg_values.append(ndcg_at_k(recs, targets, k))

        row = {
            "method": method,
            "method_label": method_labels[method],
            "K": k,
            "users_evaluated": len(test_users),
            "precision": float(np.mean(precision_values)),
            "recall": float(np.mean(recall_values)),
            "map": float(np.mean(ap_values)),
            "ndcg": float(np.mean(ndcg_values)),
        }
        metric_rows.append(row)

        print(
            f"K={k:2d} | P={row['precision']:.5f} | "
            f"R={row['recall']:.5f} | MAP={row['map']:.5f} | "
            f"NDCG={row['ndcg']:.5f}"
        )

    # Отдельная оценка новых товаров: target item должен отсутствовать
    # в истории пользователя. Считаем hits и recall именно среди новых товаров.
    total_new_targets = 0
    new_hits = 0
    users_with_20 = 0
    recommendation_counts = []
    graph_fills = 0
    popularity_fills = 0

    for user_id in test_users:
        recs = per_user_recs[user_id][:TOP_K]
        history = user_history[user_id]
        new_targets = user_new_targets[user_id]

        total_new_targets += len(new_targets)
        new_hits += sum(item in new_targets for item in recs)
        recommendation_counts.append(len(recs))

        if len(recs) == TOP_K:
            users_with_20 += 1

        primary_count = min(len(repeat_ranked[user_id]), TOP_K)
        filler_count = max(0, len(recs) - primary_count)

        if method == "repeat_plus_graph":
            graph_fills += filler_count
        elif method == "repeat_plus_popularity":
            popularity_fills += filler_count

    new_item_rows.append({
        "method": method,
        "method_label": method_labels[method],
        "K": TOP_K,
        "new_target_items_total": total_new_targets,
        "new_target_items_found": new_hits,
        "new_item_recall": (
            new_hits / total_new_targets if total_new_targets else 0.0
        ),
        "users_with_20_recommendations": users_with_20,
        "share_with_20_recommendations": users_with_20 / len(test_users),
        "mean_recommendation_count": float(np.mean(recommendation_counts)),
        "filler_recommendations": (
            graph_fills if method == "repeat_plus_graph"
            else popularity_fills if method == "repeat_plus_popularity"
            else 0
        ),
    })

metrics_df = pd.DataFrame(metric_rows)
new_items_df = pd.DataFrame(new_item_rows)

metrics_path = OUTPUT_DIR / "filling_methods_test_metrics.csv"
new_items_path = OUTPUT_DIR / "filling_methods_test_new_items.csv"

metrics_df.to_csv(metrics_path, index=False)
new_items_df.to_csv(new_items_path, index=False)

print("\n" + "=" * 70)
print("ИТОГОВОЕ СРАВНЕНИЕ МЕТРИК")
print("=" * 70)
print(metrics_df.to_string(index=False))

print("\n" + "=" * 70)
print("ОБНАРУЖЕНИЕ НОВЫХ ТОВАРОВ")
print("=" * 70)
print(new_items_df.to_string(index=False))

print("\nРезультаты сохранены:")
print(metrics_path)
print(new_items_path)
print("\nЭксперимент завершён. Все параметры и популярность рассчитаны на train;")
print("финальная оценка способов дополнения выполнена только на test.")
