
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Папка для сохранения графиков
OUTPUT_DIR = Path("data/processed/temporal_analysis/plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Настройки отображения
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.labelsize"] = 11


# ============================================================
# 1. ИСХОДНЫЕ ДАННЫЕ
# ============================================================

# Заказы по дням недели
days_data = {
    "day": ["Вс", "Пн", "Вт", "Ср", "Чт", "Пт", "Сб"],
    "orders": [
        557772, 556705, 441955, 412400,
        401212, 425982, 418848
    ],
    "reorder_rate": [
        0.585276, 0.603843, 0.589771, 0.586272,
        0.590979, 0.595470, 0.574369
    ]
}

days = pd.DataFrame(days_data)


# Заказы и повторные покупки по часам
hours_data = {
    "hour": list(range(24)),

    "orders": [
        21372, 11596, 7070, 5120, 5175, 8972,
        28792, 86656, 168321, 243496, 271885,
        268006, 256206, 261174, 265556, 266132,
        255949, 214080, 170998, 131620, 98109,
        73436, 57540, 37613
    ],

    "reorder_rate": [
        0.565500, 0.557114, 0.554455, 0.559498,
        0.571195, 0.607947, 0.636497, 0.644473,
        0.631883, 0.619449, 0.600308, 0.587324,
        0.579308, 0.579908, 0.581300, 0.580738,
        0.578629, 0.574774, 0.573437, 0.574228,
        0.582267, 0.590680, 0.585259, 0.579653
    ]
}

hours = pd.DataFrame(hours_data)



def save_plot(filename):
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / filename,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()
    print(f"Сохранен график: {filename}")



plt.figure(figsize=(12, 5))

plt.bar(hours["hour"], hours["orders"])

plt.title("Распределение заказов по часам суток")
plt.xlabel("Час суток")
plt.ylabel("Количество заказов")

plt.xticks(range(24), [f"{h:02d}:00" for h in range(24)], rotation=45)
plt.grid(axis="y", alpha=0.3)

save_plot("orders_by_hour.png")


plt.figure(figsize=(9, 5))

plt.bar(days["day"], days["orders"])

plt.title("Распределение заказов по дням недели")
plt.xlabel("День недели")
plt.ylabel("Количество заказов")

plt.grid(axis="y", alpha=0.3)

save_plot("orders_by_day.png")



plt.figure(figsize=(12, 5))

plt.plot(
    hours["hour"],
    hours["reorder_rate"] * 100,
    marker="o"
)

plt.title("Доля повторных покупок по часам суток")
plt.xlabel("Час суток")
plt.ylabel("Доля повторных покупок, %")

plt.xticks(range(24), [f"{h:02d}:00" for h in range(24)], rotation=45)
plt.grid(alpha=0.3)

save_plot("reorder_rate_by_hour.png")


plt.figure(figsize=(9, 5))

plt.bar(
    days["day"],
    days["reorder_rate"] * 100
)

plt.title("Доля повторных покупок по дням недели")
plt.xlabel("День недели")
plt.ylabel("Доля повторных покупок, %")

plt.grid(axis="y", alpha=0.3)

save_plot("reorder_rate_by_day.png")


print("\nВсе графики успешно построены!")
print(f"Папка с результатами: {OUTPUT_DIR.resolve()}")
