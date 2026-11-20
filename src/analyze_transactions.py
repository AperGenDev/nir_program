
import duckdb

DATA = "data/raw"

con = duckdb.connect()

orders_path = f"{DATA}/order_products__prior.csv"

print("=== 1. СТАТИСТИКА КОРЗИН ===")

print(con.execute(f"""
    WITH baskets AS (
        SELECT
            order_id,
            COUNT(DISTINCT product_id) AS basket_size
        FROM read_csv_auto('{orders_path}')
        GROUP BY order_id
    ),
    dataset_stats AS (
        SELECT
            COUNT(*) AS total_rows,
            COUNT(DISTINCT product_id) AS unique_products
        FROM read_csv_auto('{orders_path}')
    )
    SELECT
        dataset_stats.total_rows,
        COUNT(baskets.order_id) AS total_orders,
        dataset_stats.unique_products,
        ROUND(AVG(baskets.basket_size), 2) AS avg_basket_size,
        MEDIAN(baskets.basket_size) AS median_basket_size,
        MAX(baskets.basket_size) AS max_basket_size
    FROM baskets
    CROSS JOIN dataset_stats
    GROUP BY
        dataset_stats.total_rows,
        dataset_stats.unique_products
""").df().to_string(index=False))


print("\n=== 2. РАСПРЕДЕЛЕНИЕ РАЗМЕРА КОРЗИН ===")

print(con.execute(f"""
    SELECT
        basket_size,
        COUNT(*) AS orders
    FROM (
        SELECT
            order_id,
            COUNT(DISTINCT product_id) AS basket_size
        FROM read_csv_auto('{orders_path}')
        GROUP BY order_id
    )
    GROUP BY basket_size
    ORDER BY basket_size
    LIMIT 20
""").df().to_string(index=False))


print("\n=== 3. ЧАСТОТА ПОКУПКИ ТОВАРОВ ===")

print(con.execute(f"""
    WITH product_frequency AS (
        SELECT
            product_id,
            COUNT(DISTINCT order_id) AS order_count
        FROM read_csv_auto('{orders_path}')
        GROUP BY product_id
    )
    SELECT
        COUNT(*) AS unique_products,
        MIN(order_count) AS min_orders,
        MEDIAN(order_count) AS median_orders,
        MAX(order_count) AS max_orders,
        COUNT(*) FILTER (
            WHERE order_count >= 16000
        ) AS products_support_005,
        COUNT(*) FILTER (
            WHERE order_count >= 32000
        ) AS products_support_010,
        COUNT(*) FILTER (
            WHERE order_count >= 64000
        ) AS products_support_020
    FROM product_frequency
""").df().to_string(index=False))


print("\n=== 4. ТОП-20 ПОПУЛЯРНЫХ ТОВАРОВ ===")

print(con.execute(f"""
    SELECT
        p.product_name,
        COUNT(DISTINCT op.order_id) AS order_count
    FROM read_csv_auto('{orders_path}') op
    JOIN read_csv_auto('{DATA}/products.csv') p
        ON op.product_id = p.product_id
    GROUP BY p.product_id, p.product_name
    ORDER BY order_count DESC
    LIMIT 20
""").df().to_string(index=False))


con.close()
