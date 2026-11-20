
from pathlib import Path

import duckdb


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

RAW = Path("data/raw")
OUTPUT = Path("data/processed/repeat_prediction")

OUTPUT.mkdir(parents=True, exist_ok=True)

FILES = [
    "orders.csv",
    "order_products__prior.csv",
    "order_products__train.csv",
    "products.csv",
]

for filename in FILES:
    path = RAW / filename
    if not path.exists():
        raise FileNotFoundError(f"Не найден файл: {path}")

DB_PATH = str(OUTPUT / "preparation.duckdb")

con = duckdb.connect(DB_PATH)

# DuckDB самостоятельно выбирает план выполнения запросов.
con.execute("SET preserve_insertion_order = false")


# --------------------------------------------------
# 1. SOURCE TABLES
# --------------------------------------------------

print("1. Подключение исходных данных...")

con.execute(f"""
    CREATE OR REPLACE VIEW orders AS
    SELECT
        order_id,
        user_id,
        order_number,
        eval_set,
        order_dow,
        order_hour_of_day,
        days_since_prior_order
    FROM read_csv_auto('{(RAW / "orders.csv").as_posix()}')
""")

con.execute(f"""
    CREATE OR REPLACE VIEW prior AS
    SELECT
        order_id,
        product_id,
        add_to_cart_order,
        reordered
    FROM read_csv_auto(
        '{(RAW / "order_products__prior.csv").as_posix()}'
    )
""")

con.execute(f"""
    CREATE OR REPLACE VIEW target_products AS
    SELECT
        order_id,
        product_id
    FROM read_csv_auto(
        '{(RAW / "order_products__train.csv").as_posix()}'
    )
""")

con.execute(f"""
    CREATE OR REPLACE VIEW products AS
    SELECT
        product_id,
        aisle_id,
        department_id
    FROM read_csv_auto('{(RAW / "products.csv").as_posix()}')
""")


# --------------------------------------------------
# 2. TARGET ORDERS
# --------------------------------------------------

print("2. Формирование целевых заказов...")

con.execute("""
    CREATE OR REPLACE TABLE target_orders AS
    SELECT
        order_id,
        user_id,
        order_number,
        order_dow,
        order_hour_of_day,
        COALESCE(days_since_prior_order, 0)
            AS days_since_prior_order,
        CASE
            WHEN hash(user_id) % 10 IN (0, 5) THEN 'test'
            WHEN hash(user_id) % 10 = 1 THEN 'validation'
            ELSE 'train'
        END AS split_name

    FROM orders
    WHERE eval_set = 'train'
""")

target_users = con.execute("""
    SELECT
        split_name,
        COUNT(*) AS users_count
    FROM target_orders
    GROUP BY split_name
    ORDER BY split_name
""").df()

print("\nРазделение пользователей:")
print(target_users.to_string(index=False))


# --------------------------------------------------
# 3. USER STATISTICS
# --------------------------------------------------


print("\n3. Расчет характеристик пользователей...")

con.execute("""
    CREATE OR REPLACE TABLE user_basket_stats AS
    SELECT
        o.user_id,
        AVG(b.basket_size) AS user_avg_basket_size
    FROM (
        SELECT
            order_id,
            COUNT(*) AS basket_size
        FROM prior
        GROUP BY order_id
    ) b
    JOIN orders o
        ON b.order_id = o.order_id
    GROUP BY o.user_id
""")

con.execute("""
    CREATE OR REPLACE TABLE user_stats AS
    SELECT
        o.user_id,
        COUNT(DISTINCT p.order_id) AS user_orders_count,
        COUNT(*) AS user_total_items,
        AVG(p.add_to_cart_order) AS user_avg_cart_position,
        bs.user_avg_basket_size
    FROM prior p
    JOIN orders o
        ON p.order_id = o.order_id
    JOIN user_basket_stats bs
        ON o.user_id = bs.user_id
    GROUP BY
        o.user_id,
        bs.user_avg_basket_size
""")


# --------------------------------------------------
# 4. USER-PRODUCT STATISTICS
# --------------------------------------------------

print("4. Расчет истории покупок товаров пользователями...")

con.execute("""
    CREATE OR REPLACE TABLE user_product_stats AS
    SELECT
        o.user_id,
        p.product_id,

        COUNT(*) AS user_product_orders_count,

        MAX(o.order_number)
            AS user_product_last_order_number,

        AVG(p.reordered)
            AS user_product_reorder_rate,

        AVG(p.add_to_cart_order)
            AS user_product_avg_cart_position

    FROM prior p
    JOIN orders o
        ON p.order_id = o.order_id

    GROUP BY
        o.user_id,
        p.product_id
""")


# --------------------------------------------------
# 5. PRODUCT STATISTICS
#    Only TRAIN users contribute to global statistics.
# --------------------------------------------------

print("5. Расчет популярности товаров по обучающей части...")

con.execute("""
    CREATE OR REPLACE TABLE product_stats AS
    SELECT
        p.product_id,

        COUNT(*) AS product_prior_orders,

        COUNT(DISTINCT o.user_id)
            AS product_prior_users,

        AVG(p.reordered)
            AS product_reorder_rate

    FROM prior p
    JOIN orders o
        ON p.order_id = o.order_id

    JOIN target_orders t
        ON o.user_id = t.user_id
       AND t.split_name = 'train'

    GROUP BY p.product_id
""")


# --------------------------------------------------
# 6. BUILD CANDIDATE DATASET
# --------------------------------------------------

print("\n6. Формирование обучающей и тестовой выборок...")

con.execute("""
    CREATE OR REPLACE TABLE repeat_dataset AS

    SELECT
        t.user_id,
        t.order_id AS target_order_id,
        ups.product_id,

        t.split_name,

        -- User features
        us.user_orders_count,
        us.user_total_items,
        us.user_avg_cart_position,
        us.user_avg_basket_size,

        -- User-product features
        ups.user_product_orders_count,
        ups.user_product_last_order_number,

        t.order_number - ups.user_product_last_order_number
            AS user_product_recency,

        ups.user_product_reorder_rate,
        ups.user_product_avg_cart_position,

        -- Product features
        COALESCE(ps.product_prior_orders, 0)
            AS product_prior_orders,

        COALESCE(ps.product_prior_users, 0)
            AS product_prior_users,

        COALESCE(ps.product_reorder_rate, 0)
            AS product_reorder_rate,

        -- Temporal features
        t.order_dow,
        t.order_hour_of_day,
        t.days_since_prior_order,

        -- Product category
        pr.aisle_id,
        pr.department_id,

        -- Target
        CASE
            WHEN tp.product_id IS NOT NULL THEN 1
            ELSE 0
        END AS target

    FROM user_product_stats ups

    JOIN target_orders t
        ON ups.user_id = t.user_id

    JOIN user_stats us
        ON ups.user_id = us.user_id

    LEFT JOIN product_stats ps
        ON ups.product_id = ps.product_id

    LEFT JOIN products pr
        ON ups.product_id = pr.product_id

    LEFT JOIN target_products tp
        ON tp.order_id = t.order_id
       AND tp.product_id = ups.product_id
""")


# --------------------------------------------------
# 7. SAVE TRAIN AND TEST
# --------------------------------------------------

print("7. Сохранение выборок...")

for split_name in ["train", "validation", "test"]:
    output_file = (
        OUTPUT / f"{split_name}_repeat.parquet"
    ).as_posix()

    con.execute(f"""
        COPY (
            SELECT *
            FROM repeat_dataset
            WHERE split_name = '{split_name}'
        )
        TO '{output_file}'
        (FORMAT PARQUET, COMPRESSION ZSTD)
    """)


# --------------------------------------------------
# 8. DATASET STATISTICS
# --------------------------------------------------

print("\n8. Итоговая статистика...")

stats = con.execute("""
    SELECT
        split_name,
        COUNT(*) AS candidate_rows,
        COUNT(DISTINCT user_id) AS users_count,
        COUNT(DISTINCT product_id) AS products_count,
        SUM(target) AS positive_examples,
        AVG(target) AS positive_rate
    FROM repeat_dataset
    GROUP BY split_name
    ORDER BY split_name
""").df()

print("\nСтатистика выборок:")
print(stats.to_string(index=False))


# --------------------------------------------------
# 9. TARGET COVERAGE
# --------------------------------------------------

coverage = con.execute("""
    SELECT
        t.split_name,

        COUNT(*) AS target_items,

        COUNT(*) FILTER (
            WHERE ups.product_id IS NOT NULL
        ) AS items_seen_before,

        COUNT(*) FILTER (
            WHERE ups.product_id IS NULL
        ) AS unseen_items,

        COUNT(*) FILTER (
            WHERE ups.product_id IS NOT NULL
        ) * 1.0 / COUNT(*) AS candidate_coverage

    FROM target_orders t

    JOIN target_products tp
        ON t.order_id = tp.order_id

    LEFT JOIN user_product_stats ups
        ON t.user_id = ups.user_id
       AND tp.product_id = ups.product_id

    GROUP BY t.split_name
    ORDER BY t.split_name
""").df()

print("\nПокрытие целевых товаров историей:")
print(coverage.to_string(index=False))

coverage.to_csv(
    OUTPUT / "target_coverage.csv",
    index=False
)

stats.to_csv(
    OUTPUT / "dataset_statistics.csv",
    index=False
)

print("\nГотово!")
print(f"Результаты сохранены в: {OUTPUT}")

con.close()
