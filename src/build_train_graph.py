
from pathlib import Path

import duckdb
import pandas as pd
import networkx as nx


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

RAW = ROOT / "data" / "raw"
OUTPUT = ROOT / "data" / "processed" / "graph_experiments"

OUTPUT.mkdir(parents=True, exist_ok=True)

# Все конфигурации строятся на train-пользователях по тому же split,
# что и repeat_prediction: test=(0,5), validation=1, train=остальные.
GRAPH_CONFIGS = [
    {"top_products": 252,  "min_pair_orders": 500, "min_lift": 1.2},
    {"top_products": 500,  "min_pair_orders": 300, "min_lift": 1.2},
    {"top_products": 1000, "min_pair_orders": 200, "min_lift": 1.2},
    {"top_products": 2000, "min_pair_orders": 100, "min_lift": 1.2},
]

DB_PATH = OUTPUT / "graph_experiments.duckdb"

con = duckdb.connect(str(DB_PATH))
con.execute("SET preserve_insertion_order = false")


def sql_path(path):
    return path.as_posix().replace("'", "''")


# ============================================================
# 1. LOAD DATA
# ============================================================

print("1. Подключение исходных данных...")

con.execute(f"""
    CREATE OR REPLACE VIEW orders AS
    SELECT order_id, user_id, eval_set
    FROM read_csv_auto('{sql_path(RAW / "orders.csv")}')
""")

con.execute(f"""
    CREATE OR REPLACE VIEW prior AS
    SELECT order_id, product_id
    FROM read_csv_auto('{sql_path(RAW / "order_products__prior.csv")}')
""")

con.execute(f"""
    CREATE OR REPLACE VIEW products AS
    SELECT product_id, product_name
    FROM read_csv_auto('{sql_path(RAW / "products.csv")}')
""")


# ============================================================
# 2. TRAIN USERS
# ============================================================

print("\n2. Выделение обучающих пользователей...")

con.execute("""
    CREATE OR REPLACE TABLE train_users AS
    SELECT DISTINCT user_id
    FROM orders
    WHERE eval_set = 'train'
      AND hash(user_id) % 10 NOT IN (0, 1, 5)
""")

train_users_count = con.execute("""
    SELECT COUNT(*) FROM train_users
""").fetchone()[0]

print(f"Обучающих пользователей: {train_users_count:,}")


# ============================================================
# 3. TRAIN TRANSACTIONS
# ============================================================

print("\n3. Формирование обучающих транзакций...")

con.execute("""
    CREATE OR REPLACE TABLE train_transactions AS
    SELECT
        p.order_id,
        p.product_id
    FROM prior p
    JOIN orders o ON p.order_id = o.order_id
    JOIN train_users u ON o.user_id = u.user_id
""")

total_orders = con.execute("""
    SELECT COUNT(DISTINCT order_id)
    FROM train_transactions
""").fetchone()[0]

transaction_stats = con.execute("""
    SELECT
        COUNT(*) AS total_items,
        COUNT(DISTINCT order_id) AS total_orders,
        COUNT(DISTINCT product_id) AS unique_products
    FROM train_transactions
""").df()

print(transaction_stats.to_string(index=False))


# ============================================================
# 4. BUILD EACH GRAPH CONFIGURATION
# ============================================================

all_statistics = []

for config in GRAPH_CONFIGS:

    top_products = config["top_products"]
    min_pair_orders = config["min_pair_orders"]
    min_lift = config["min_lift"]

    config_name = (
        f"top_{top_products}_"
        f"pair_{min_pair_orders}_"
        f"lift_{min_lift}"
    )

    config_dir = OUTPUT / config_name
    config_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 65)
    print(f"КОНФИГУРАЦИЯ: {config_name}")
    print("=" * 65)

    # --------------------------------------------------------
    # 4.1 TOP PRODUCTS
    # --------------------------------------------------------

    print("\n4.1. Выбор популярных товаров...")

    con.execute(f"""
        CREATE OR REPLACE TABLE graph_nodes AS
        SELECT
            t.product_id,
            p.product_name,
            COUNT(DISTINCT t.order_id) AS item_orders
        FROM train_transactions t
        JOIN products p ON t.product_id = p.product_id
        GROUP BY t.product_id, p.product_name
        ORDER BY item_orders DESC
        LIMIT {top_products}
    """)

    nodes_count = con.execute("""
        SELECT COUNT(*) FROM graph_nodes
    """).fetchone()[0]

    print(f"Выбрано товаров: {nodes_count:,}")

    # --------------------------------------------------------
    # 4.2 FILTER TRANSACTIONS
    # --------------------------------------------------------

    print("4.2. Фильтрация транзакций...")

    con.execute("""
        CREATE OR REPLACE TABLE selected_transactions AS
        SELECT
            t.order_id,
            t.product_id
        FROM train_transactions t
        JOIN graph_nodes n ON t.product_id = n.product_id
    """)

    # --------------------------------------------------------
    # 4.3 ITEM COUNTS
    # --------------------------------------------------------

    print("4.3. Расчет частот товаров...")

    con.execute("""
        CREATE OR REPLACE TABLE item_counts AS
        SELECT
            product_id,
            COUNT(DISTINCT order_id) AS item_orders
        FROM selected_transactions
        GROUP BY product_id
    """)

    # --------------------------------------------------------
    # 4.4 PAIR COUNTS
    # --------------------------------------------------------

    print("4.4. Расчет совместных покупок...")

    con.execute("""
        CREATE OR REPLACE TABLE pair_counts AS
        SELECT
            a.product_id AS product_a,
            b.product_id AS product_b,
            COUNT(*) AS pair_orders
        FROM selected_transactions a
        JOIN selected_transactions b
            ON a.order_id = b.order_id
           AND a.product_id < b.product_id
        GROUP BY a.product_id, b.product_id
    """)

    pair_count = con.execute("""
        SELECT COUNT(*) FROM pair_counts
    """).fetchone()[0]

    print(f"Всего пар до фильтрации: {pair_count:,}")

    # --------------------------------------------------------
    # 4.5 ASSOCIATION METRICS
    # --------------------------------------------------------

    print("4.5. Расчет Support, Confidence и Lift...")

    con.execute(f"""
        CREATE OR REPLACE TABLE graph_edges AS
        SELECT
            pc.product_a,
            pc.product_b,
            pc.pair_orders,

            pc.pair_orders * 1.0 / {total_orders}
                AS support,

            pc.pair_orders * 1.0 / a.item_orders
                AS confidence_a_to_b,

            pc.pair_orders * 1.0 / b.item_orders
                AS confidence_b_to_a,

            (
                pc.pair_orders * {total_orders} * 1.0
                / (a.item_orders * b.item_orders)
            ) AS lift

        FROM pair_counts pc

        JOIN item_counts a
            ON pc.product_a = a.product_id

        JOIN item_counts b
            ON pc.product_b = b.product_id

        WHERE pc.pair_orders >= {min_pair_orders}
          AND (
              pc.pair_orders * {total_orders} * 1.0
              / (a.item_orders * b.item_orders)
          ) >= {min_lift}
    """)

    nodes = con.execute("""
        SELECT * FROM graph_nodes
        ORDER BY item_orders DESC
    """).df()

    edges = con.execute("""
        SELECT * FROM graph_edges
        ORDER BY pair_orders DESC
    """).df()

    # --------------------------------------------------------
    # 4.6 NETWORKX GRAPH
    # --------------------------------------------------------

    print("4.6. Построение графа...")

    G = nx.Graph()

    for row in nodes.itertuples(index=False):
        G.add_node(
            int(row.product_id),
            product_name=str(row.product_name),
            item_orders=int(row.item_orders)
        )

    for row in edges.itertuples(index=False):
        G.add_edge(
            int(row.product_a),
            int(row.product_b),
            pair_orders=int(row.pair_orders),
            support=float(row.support),
            confidence_a_to_b=float(row.confidence_a_to_b),
            confidence_b_to_a=float(row.confidence_b_to_a),
            lift=float(row.lift)
        )

    components = list(nx.connected_components(G))

    largest_component = (
        max(map(len, components)) if components else 0
    )

    degrees = dict(G.degree())

    stats = {
        "config": config_name,
        "train_users": train_users_count,
        "train_orders": total_orders,
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "components": len(components),
        "largest_component": largest_component,
        "density": nx.density(G),
        "avg_degree": (
            sum(degrees.values()) / len(degrees)
            if degrees else 0
        ),
        "min_pair_orders": min_pair_orders,
        "min_lift": min_lift
    }

    all_statistics.append(stats)

    # --------------------------------------------------------
    # 4.7 SAVE RESULTS
    # --------------------------------------------------------

    print("4.7. Сохранение результатов...")

    nodes.to_csv(config_dir / "graph_nodes.csv", index=False)
    edges.to_csv(config_dir / "graph_edges.csv", index=False)

    pd.DataFrame([stats]).to_csv(
        config_dir / "graph_statistics.csv",
        index=False
    )

    nx.write_graphml(
        G,
        config_dir / "product_graph.graphml"
    )

    degree_df = pd.DataFrame([
        {
            "product_id": node,
            "product_name": G.nodes[node]["product_name"],
            "degree": degree,
            "item_orders": G.nodes[node]["item_orders"]
        }
        for node, degree in G.degree()
    ]).sort_values("degree", ascending=False)

    degree_df.to_csv(
        config_dir / "products_by_degree.csv",
        index=False
    )

    print("\n=== СТАТИСТИКА ГРАФА ===")
    print(pd.DataFrame([stats]).to_string(index=False))

    print("\nТОП-10 ТОВАРОВ ПО ЧИСЛУ СВЯЗЕЙ")
    print(degree_df.head(10).to_string(index=False))

    # Удаляем промежуточные таблицы, чтобы не накапливать
    # большие результаты между экспериментами.
    con.execute("DROP TABLE IF EXISTS selected_transactions")
    con.execute("DROP TABLE IF EXISTS item_counts")
    con.execute("DROP TABLE IF EXISTS pair_counts")
    con.execute("DROP TABLE IF EXISTS graph_nodes")
    con.execute("DROP TABLE IF EXISTS graph_edges")


# ============================================================
# 5. SUMMARY
# ============================================================

summary_df = pd.DataFrame(all_statistics)

summary_df.to_csv(
    OUTPUT / "graph_build_summary.csv",
    index=False
)

print("\n" + "=" * 65)
print("СВОДКА ПО ВСЕМ КОНФИГУРАЦИЯМ")
print("=" * 65)

print(summary_df.to_string(index=False))

print("\nВсе графы построены.")
print(f"Результаты: {OUTPUT}")

con.close()
