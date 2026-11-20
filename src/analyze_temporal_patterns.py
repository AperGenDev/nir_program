
import duckdb
from pathlib import Path

DATA = "data/raw"
OUTPUT = Path("data/processed/temporal_analysis")

OUTPUT.mkdir(parents=True, exist_ok=True)

con = duckdb.connect()

# 1. Загружаем заказы
con.execute(f"""
    CREATE VIEW orders AS
    SELECT
        order_id,
        user_id,
        order_number,
        order_dow,
        order_hour_of_day,
        days_since_prior_order
    FROM read_csv_auto('{DATA}/orders.csv')
""")

# 2. Загружаем товарные позиции prior
con.execute(f"""
    CREATE VIEW order_products AS
    SELECT
        order_id,
        product_id,
        reordered
    FROM read_csv_auto('{DATA}/order_products__prior.csv')
""")

# 3. Получаем только заказы, присутствующие в prior
con.execute("""
    CREATE VIEW prior_orders AS
    SELECT
        o.order_id,
        o.user_id,
        o.order_number,
        o.order_dow,
        o.order_hour_of_day,
        o.days_since_prior_order
    FROM orders o
    INNER JOIN (
        SELECT DISTINCT order_id
        FROM order_products
    ) p
        ON o.order_id = p.order_id
""")

# 4. Распределение заказов по дням недели
dow = con.execute("""
    SELECT
        order_dow,
        COUNT(*) AS orders_count,
        COUNT(DISTINCT user_id) AS unique_users
    FROM prior_orders
    GROUP BY order_dow
    ORDER BY order_dow
""").df()

dow.to_csv(OUTPUT / "orders_by_dow.csv", index=False)

print("\n=== ЗАКАЗЫ ПО ДНЯМ НЕДЕЛИ ===")
print(dow.to_string(index=False))

# 5. Распределение заказов по часам
hour = con.execute("""
    SELECT
        order_hour_of_day,
        COUNT(*) AS orders_count
    FROM prior_orders
    GROUP BY order_hour_of_day
    ORDER BY order_hour_of_day
""").df()

hour.to_csv(OUTPUT / "orders_by_hour.csv", index=False)

print("\n=== ЗАКАЗЫ ПО ЧАСАМ ===")
print(hour.to_string(index=False))

# 6. Распределение заказов по дням и часам
dow_hour = con.execute("""
    SELECT
        order_dow,
        order_hour_of_day,
        COUNT(*) AS orders_count
    FROM prior_orders
    GROUP BY order_dow, order_hour_of_day
    ORDER BY order_dow, order_hour_of_day
""").df()

dow_hour.to_csv(
    OUTPUT / "orders_by_dow_hour.csv",
    index=False
)

# 7. Повторные покупки по дням недели
dow_reorders = con.execute("""
    SELECT
        o.order_dow,
        COUNT(*) AS total_items,
        SUM(p.reordered) AS reordered_items,
        AVG(p.reordered) AS reorder_rate
    FROM order_products p
    JOIN prior_orders o
        ON p.order_id = o.order_id
    GROUP BY o.order_dow
    ORDER BY o.order_dow
""").df()

dow_reorders.to_csv(
    OUTPUT / "reorder_rate_by_dow.csv",
    index=False
)

print("\n=== ПОВТОРНЫЕ ПОКУПКИ ПО ДНЯМ НЕДЕЛИ ===")
print(dow_reorders.to_string(index=False))

# 8. Повторные покупки по часам суток
hour_reorders = con.execute("""
    SELECT
        o.order_hour_of_day,
        COUNT(*) AS total_items,
        SUM(p.reordered) AS reordered_items,
        AVG(p.reordered) AS reorder_rate
    FROM order_products p
    JOIN prior_orders o
        ON p.order_id = o.order_id
    GROUP BY o.order_hour_of_day
    ORDER BY o.order_hour_of_day
""").df()

hour_reorders.to_csv(
    OUTPUT / "reorder_rate_by_hour.csv",
    index=False
)

print("\n=== ПОВТОРНЫЕ ПОКУПКИ ПО ЧАСАМ ===")
print(hour_reorders.to_string(index=False))

# 9. Повторные покупки по дням недели и часам
reorders_dow_hour = con.execute("""
    SELECT
        o.order_dow,
        o.order_hour_of_day,
        COUNT(*) AS total_items,
        SUM(p.reordered) AS reordered_items,
        AVG(p.reordered) AS reorder_rate
    FROM order_products p
    JOIN prior_orders o
        ON p.order_id = o.order_id
    GROUP BY
        o.order_dow,
        o.order_hour_of_day
    ORDER BY
        o.order_dow,
        o.order_hour_of_day
""").df()

reorders_dow_hour.to_csv(
    OUTPUT / "reorder_rate_by_dow_hour.csv",
    index=False
)

print("\n=== РЕЗУЛЬТАТЫ СОХРАНЕНЫ ===")
print(OUTPUT)

con.close()

print("\nИсследование завершено.")
