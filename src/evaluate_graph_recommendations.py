
from pathlib import Path
import math

import duckdb
import pandas as pd
import numpy as np


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw"
REPEAT_DIR = ROOT / "data" / "processed" / "repeat_prediction"

GRAPH_ROOT = ROOT / "data" / "processed" / "graph_experiments"

VALIDATION_FILE = REPEAT_DIR / "validation_repeat.parquet"
TRAIN_PRODUCTS_FILE = RAW_DIR / "order_products__train.csv"

OUTPUT_DIR = (
    ROOT / "data" / "processed" / "graph_experiment_evaluation"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

K_VALUES = [5, 10, 20]

GRAPH_CONFIGS = [
    "top_252_pair_500_lift_1.2",
    "top_500_pair_300_lift_1.2",
    "top_1000_pair_200_lift_1.2",
    "top_2000_pair_100_lift_1.2",
]


# ============================================================
# METRICS
# ============================================================

def precision_at_k(recommended, relevant, k):
    """
    Precision@K с фиксированным знаменателем K.

    Если модель выдала меньше K рекомендаций,
    недостающие позиции считаются непопаданиями.
    """
    hits = sum(item in relevant for item in recommended[:k])
    return hits / k


def recall_at_k(recommended, relevant, k):
    if not relevant:
        return 0.0

    hits = sum(item in relevant for item in recommended[:k])

    return hits / len(relevant)


def average_precision_at_k(recommended, relevant, k):
    """
    AP@K нормируется относительно полного множества
    релевантных товаров, а не только товаров из графа.
    """
    if not relevant:
        return 0.0

    score = 0.0
    hits = 0

    for rank, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            hits += 1
            score += hits / rank

    return score / min(len(relevant), k)


def ndcg_at_k(recommended, relevant, k):
    if not relevant:
        return 0.0

    dcg = 0.0

    for rank, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            dcg += 1.0 / math.log2(rank + 1)

    # Идеальное ранжирование строится по полному
    # множеству релевантных товаров пользователя.
    ideal_hits = min(len(relevant), k)

    idcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_hits + 1)
    )

    return dcg / idcg if idcg > 0 else 0.0


# ============================================================
# 1. LOAD TEST HISTORY
# ============================================================

print("1. Загрузка истории тестовых пользователей...")

con = duckdb.connect()

test_path = VALIDATION_FILE.as_posix().replace("'", "''")
target_path = TRAIN_PRODUCTS_FILE.as_posix().replace("'", "''")

history_df = con.execute(f"""
    SELECT
        user_id,
        product_id
    FROM read_parquet('{test_path}')
    GROUP BY user_id, product_id
""").df()

history_df["user_id"] = history_df["user_id"].astype(int)
history_df["product_id"] = history_df["product_id"].astype(int)

user_history = (
    history_df
    .groupby("user_id")["product_id"]
    .apply(set)
    .to_dict()
)

print(f"Тестовых пользователей: {len(user_history):,}")


# ============================================================
# 2. LOAD ALL TARGET PURCHASES
# ============================================================

print("\n2. Загрузка полных целевых корзин...")

test_orders = con.execute(f"""
    SELECT DISTINCT
        user_id,
        target_order_id
    FROM read_parquet('{test_path}')
""").df()

test_orders["user_id"] = test_orders["user_id"].astype(int)
test_orders["target_order_id"] = (
    test_orders["target_order_id"].astype(int)
)

if test_orders["user_id"].duplicated().any():
    raise ValueError(
        "У пользователя найдено несколько target_order_id."
    )

con.register("test_orders_df", test_orders)

target_df = con.execute(f"""
    SELECT
        t.user_id,
        p.product_id
    FROM test_orders_df t
    JOIN read_csv_auto(
        '{target_path}',
        header = true
    ) p
        ON t.target_order_id = p.order_id
    GROUP BY
        t.user_id,
        p.product_id
""").df()

target_df["user_id"] = target_df["user_id"].astype(int)
target_df["product_id"] = target_df["product_id"].astype(int)

user_targets = (
    target_df
    .groupby("user_id")["product_id"]
    .apply(set)
    .to_dict()
)

con.close()

# Проверяем, что все тестовые пользователи имеют целевые корзины.
missing_targets = set(user_history) - set(user_targets)

if missing_targets:
    raise ValueError(
        f"У {len(missing_targets)} пользователей "
        "не найдены целевые покупки."
    )

all_target_items = set(target_df["product_id"].unique())

print(f"Пользователей с целевыми покупками: {len(user_targets):,}")
print(f"Уникальных целевых товаров: {len(all_target_items):,}")


# ============================================================
# 3. EVALUATE ONE GRAPH
# ============================================================

def evaluate_graph(config_name):

    print("\n" + "=" * 70)
    print(f"ОЦЕНКА ГРАФА: {config_name}")
    print("=" * 70)

    graph_dir = GRAPH_ROOT / config_name

    nodes_file = graph_dir / "graph_nodes.csv"
    edges_file = graph_dir / "graph_edges.csv"

    if not nodes_file.exists():
        raise FileNotFoundError(nodes_file)

    if not edges_file.exists():
        raise FileNotFoundError(edges_file)

    nodes_df = pd.read_csv(nodes_file)
    edges_df = pd.read_csv(edges_file)

    graph_products = set(
        nodes_df["product_id"].astype(int)
    )

    # --------------------------------------------------------
    # BUILD ADJACENCY
    # --------------------------------------------------------

    adjacency = {
        product_id: {}
        for product_id in graph_products
    }

    for row in edges_df.itertuples(index=False):

        a = int(row.product_a)
        b = int(row.product_b)

        conf_ab = float(row.confidence_a_to_b)
        conf_ba = float(row.confidence_b_to_a)

        adjacency[a][b] = conf_ab
        adjacency[b][a] = conf_ba

    print(f"Вершин: {len(graph_products):,}")
    print(f"Рёбер: {len(edges_df):,}")

    # --------------------------------------------------------
    # TARGET COVERAGE
    # --------------------------------------------------------

    graph_target_items = all_target_items & graph_products

    target_catalog_coverage = (
        len(graph_target_items) / len(all_target_items)
        if all_target_items else 0.0
    )

    # --------------------------------------------------------
    # GENERATE RECOMMENDATIONS
    # --------------------------------------------------------

    print("Формирование рекомендаций...")

    recommendations = {}

    for user_id, history in user_history.items():

        scores = {}

        # Каждый товар из истории выступает исходной вершиной.
        for product_id in history:

            neighbors = adjacency.get(product_id, {})

            for candidate_id, confidence in neighbors.items():

                # Исключаем товары, уже купленные пользователем.
                if candidate_id in history:
                    continue

                scores[candidate_id] = (
                    scores.get(candidate_id, 0.0) + confidence
                )

        # Стабильная сортировка при одинаковых оценках.
        ranked = sorted(
            scores.items(),
            key=lambda x: (-x[1], x[0])
        )

        recommendations[user_id] = [
            product_id
            for product_id, score in ranked
        ]

    print("Рекомендации сформированы.")

    # --------------------------------------------------------
    # USER GROUPS
    # --------------------------------------------------------

    all_users = list(user_history.keys())

    eligible_users = [
        user_id
        for user_id in all_users
        if user_targets[user_id] & graph_products
    ]

    print(
        "Пользователей с целевыми товарами в графе:",
        f"{len(eligible_users):,}"
    )

    # --------------------------------------------------------
    # TWO EVALUATION SCOPES
    # --------------------------------------------------------

    evaluation_scopes = {
        "end_to_end": all_users,
        "conditional_graph_catalog": eligible_users
    }

    all_metrics = []

    for scope_name, users in evaluation_scopes.items():

        print(f"\nОценка: {scope_name}")
        print(f"Количество пользователей: {len(users):,}")

        for k in K_VALUES:

            precisions = []
            recalls = []
            aps = []
            ndcgs = []

            recommended_items = set()
            recommendation_counts = []

            for user_id in users:

                full_relevant = user_targets[user_id]

                if scope_name == "end_to_end":
                    relevant = full_relevant
                else:
                    relevant = full_relevant & graph_products

                ranked = recommendations.get(user_id, [])
                top_k = ranked[:k]

                recommended_items.update(top_k)
                recommendation_counts.append(len(top_k))

                precisions.append(
                    precision_at_k(ranked, relevant, k)
                )

                recalls.append(
                    recall_at_k(ranked, relevant, k)
                )

                aps.append(
                    average_precision_at_k(ranked, relevant, k)
                )

                ndcgs.append(
                    ndcg_at_k(ranked, relevant, k)
                )

            metrics = {
                "config": config_name,
                "scope": scope_name,
                "K": k,
                "users_evaluated": len(users),

                "precision": (
                    np.mean(precisions) if precisions else 0.0
                ),

                "recall": (
                    np.mean(recalls) if recalls else 0.0
                ),

                "map": (
                    np.mean(aps) if aps else 0.0
                ),

                "ndcg": (
                    np.mean(ndcgs) if ndcgs else 0.0
                ),

                "catalog_coverage": (
                    len(recommended_items) / len(graph_products)
                    if graph_products else 0.0
                ),

                "unique_recommended_products": len(recommended_items),

                "users_with_recommendations": sum(
                    count > 0 for count in recommendation_counts
                ),

                "users_with_at_least_k_recommendations": sum(
                    count >= k for count in recommendation_counts
                ),

                "mean_recommendation_count": (
                    np.mean(recommendation_counts)
                    if recommendation_counts else 0.0
                )
            }

            all_metrics.append(metrics)

    # --------------------------------------------------------
    # COVERAGE STATISTICS
    # --------------------------------------------------------

    coverage = {
        "config": config_name,

        "graph_nodes": len(graph_products),
        "graph_edges": len(edges_df),

        "unique_target_items": len(all_target_items),
        "target_items_in_graph": len(graph_target_items),

        "target_catalog_coverage": target_catalog_coverage,

        "test_users": len(all_users),

        "users_with_graph_targets": len(eligible_users),

        "users_with_graph_targets_share": (
            len(eligible_users) / len(all_users)
            if all_users else 0.0
        )
    }

    metrics_df = pd.DataFrame(all_metrics)

    print("\n=== МЕТРИКИ ===")
    print(metrics_df.to_string(index=False))

    print("\n=== ПОКРЫТИЕ ===")
    print(pd.DataFrame([coverage]).to_string(index=False))

    return all_metrics, coverage


# ============================================================
# 4. RUN ALL CONFIGURATIONS
# ============================================================

all_metrics = []
all_coverage = []

for config_name in GRAPH_CONFIGS:

    metrics, coverage = evaluate_graph(config_name)

    all_metrics.extend(metrics)
    all_coverage.append(coverage)


# ============================================================
# 5. SAVE RESULTS
# ============================================================

metrics_df = pd.DataFrame(all_metrics)
coverage_df = pd.DataFrame(all_coverage)

metrics_path = OUTPUT_DIR / "graph_end_to_end_metrics.csv"
coverage_path = OUTPUT_DIR / "graph_end_to_end_coverage.csv"

metrics_df.to_csv(metrics_path, index=False)
coverage_df.to_csv(coverage_path, index=False)

# ------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------

summary_df = metrics_df.pivot(
    index=["config", "scope"],
    columns="K",
    values=[
        "precision",
        "recall",
        "map",
        "ndcg",
        "catalog_coverage",
        "mean_recommendation_count"
    ]
)

summary_df.columns = [
    f"{metric}_at_{k}"
    for metric, k in summary_df.columns
]

summary_df = summary_df.reset_index()

summary_df = summary_df.merge(
    coverage_df,
    on="config",
    how="left"
)

summary_path = OUTPUT_DIR / "graph_end_to_end_summary.csv"

summary_df.to_csv(summary_path, index=False)


# ============================================================
# 6. PRINT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("ИТОГОВАЯ СВОДКА")
print("=" * 70)

print(summary_df.to_string(index=False))

print("\nРезультаты сохранены:")
print(metrics_path)
print(coverage_path)
print(summary_path)
