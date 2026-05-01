"""
Генераторы данных для 10 магазинов.
Каждый магазин имеет своё имя, специализацию, объём и качество данных.
"""

import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Tuple


# ─── Справочники ─────────────────────────────────────────────────────────────

SHOPS: List[Dict] = [
    {"id": 0, "name": "МегаМаркет",       "specialization": "Электроника",       "emoji": "💻"},
    {"id": 1, "name": "ТехноПлаза",       "specialization": "Бытовая техника",   "emoji": "🏠"},
    {"id": 2, "name": "ГиперМолл",        "specialization": "Одежда и обувь",    "emoji": "👗"},
    {"id": 3, "name": "СупермаркетПлюс",  "specialization": "Продукты питания",  "emoji": "🛒"},
    {"id": 4, "name": "ЭкспрессТорг",    "specialization": "Косметика",         "emoji": "💄"},
    {"id": 5, "name": "ЦифраМир",        "specialization": "Программное ПО",    "emoji": "💾"},
    {"id": 6, "name": "ФудСити",         "specialization": "Рестораны/Доставка","emoji": "🍕"},
    {"id": 7, "name": "ОптТрейд",        "specialization": "Оптовая торговля",  "emoji": "📦"},
    {"id": 8, "name": "НовоТорг",        "specialization": "Спортивные товары", "emoji": "⚽"},
    {"id": 9, "name": "УниверМаг",       "specialization": "Универмаг",         "emoji": "🏪"},
]

PRODUCTS_BY_CATEGORY: Dict[str, List[Tuple[str, float, float]]] = {
    "Электроника":       [("Ноутбук", 30000, 120000), ("Смартфон", 8000, 80000),
                          ("Планшет", 12000, 60000), ("Наушники", 1500, 25000),
                          ("Монитор", 8000, 50000), ("Клавиатура", 500, 8000)],
    "Бытовая техника":   [("Холодильник", 20000, 90000), ("Стиральная машина", 15000, 60000),
                          ("Пылесос", 3000, 25000), ("Микроволновка", 3000, 15000),
                          ("Утюг", 800, 5000), ("Кофемашина", 5000, 40000)],
    "Одежда и обувь":    [("Куртка", 2000, 15000), ("Брюки", 800, 8000),
                          ("Рубашка", 500, 5000), ("Платье", 1500, 12000),
                          ("Кроссовки", 2000, 18000), ("Пальто", 4000, 25000)],
    "Продукты питания":  [("Хлеб", 30, 120), ("Молоко", 60, 180),
                          ("Яйца (10 шт)", 80, 200), ("Сыр", 200, 1200),
                          ("Масло", 90, 350), ("Йогурт", 50, 200)],
    "Косметика":         [("Крем для лица", 300, 5000), ("Шампунь", 150, 800),
                          ("Духи", 800, 15000), ("Тушь", 200, 3000),
                          ("Помада", 150, 3500), ("Тоник", 250, 2500)],
    "Программное ПО":    [("Антивирус (1 год)", 800, 3000), ("Офис 365", 3000, 8000),
                          ("VPN-подписка", 500, 2500), ("Adobe CC", 5000, 15000),
                          ("Игра AAA", 1500, 5000), ("IDE лицензия", 3000, 12000)],
    "Рестораны/Доставка":[("Пицца", 400, 1200), ("Суши-сет", 800, 2500),
                          ("Бизнес-ланч", 250, 600), ("Бургер-сет", 350, 900),
                          ("Десерт", 150, 500), ("Напиток", 80, 400)],
    "Оптовая торговля":  [("Партия товаров (кор.)", 5000, 50000), ("Упаковка (1000 шт)", 1000, 8000),
                          ("Паллет смешанный", 15000, 80000), ("Хозтовары (опт)", 3000, 20000)],
    "Спортивные товары": [("Велосипед", 8000, 50000), ("Гантели", 500, 5000),
                          ("Беговая дорожка", 15000, 80000), ("Коврик", 300, 2000),
                          ("Перчатки боксёрские", 800, 4000), ("Ракетка", 1000, 15000)],
    "Универмаг":         [("Постельное бельё", 1500, 8000), ("Посуда (набор)", 2000, 12000),
                          ("Детская игрушка", 300, 5000), ("Книга", 200, 1500),
                          ("Канцтовары (набор)", 150, 800), ("Декор для дома", 500, 8000)],
}

PROMOTIONS = ["Распродажа -20%", "2+1 бесплатно", "Скидка дня", "Без скидки", "Без скидки", "Без скидки"]


# ─── Модели данных ────────────────────────────────────────────────────────────

@dataclass
class Product:
    name: str
    category: str
    price: float
    original_price: float
    quantity: int
    in_stock: bool
    promotion: str
    rating: float
    timestamp: datetime
    shop_id: int

    @property
    def discount_pct(self) -> float:
        if self.original_price > 0:
            return round((1 - self.price / self.original_price) * 100, 1)
        return 0.0

    @property
    def total_value(self) -> float:
        return self.price * self.quantity


@dataclass
class ShopDataBatch:
    shop_id: int
    shop_name: str
    specialization: str
    emoji: str
    products: List[Product]
    timestamp: datetime
    data_quality: float          # 0.0–1.0
    transmission_delay_ms: float # симулируем задержку сети

    @property
    def batch_size(self) -> int:
        return len(self.products)

    @property
    def avg_price(self) -> float:
        if not self.products:
            return 0.0
        return round(sum(p.price for p in self.products) / len(self.products), 2)

    @property
    def total_value(self) -> float:
        return round(sum(p.total_value for p in self.products), 2)

    @property
    def unique_categories(self) -> int:
        return len(set(p.category for p in self.products))

    @property
    def out_of_stock_count(self) -> int:
        return sum(1 for p in self.products if not p.in_stock)

    def to_stats_dict(self) -> Dict:
        return {
            "Магазин": f"{self.emoji} {self.shop_name}",
            "Специализация": self.specialization,
            "Товаров в партии": self.batch_size,
            "Ср. цена (₽)": self.avg_price,
            "Оборот (₽)": self.total_value,
            "Категорий": self.unique_categories,
            "Не в наличии": self.out_of_stock_count,
            "Качество данных": f"{self.data_quality:.0%}",
            "Задержка (мс)": round(self.transmission_delay_ms, 1),
        }


# ─── Генератор данных одного магазина ────────────────────────────────────────

class ShopDataGenerator:
    """Генерирует случайные пакеты данных для одного магазина."""

    def __init__(self, shop_id: int, seed: int | None = None):
        info = SHOPS[shop_id]
        self.shop_id = info["id"]
        self.shop_name = info["name"]
        self.specialization = info["specialization"]
        self.emoji = info["emoji"]

        if seed is not None:
            random.seed(seed + shop_id)

        # Индивидуальные характеристики магазина
        self.price_multiplier = random.uniform(0.75, 1.40)
        self.volume_level = random.choice(["low", "low", "medium", "medium", "medium", "high"])
        self.base_quality = random.uniform(0.55, 0.98)
        self.network_stability = random.uniform(0.7, 1.0)  # влияет на задержку

    # Объём пакета в зависимости от уровня
    _VOLUME_RANGE = {"low": (5, 15), "medium": (16, 40), "high": (41, 80)}

    def generate_batch(self) -> ShopDataBatch:
        lo, hi = self._VOLUME_RANGE[self.volume_level]
        n = random.randint(lo, hi)

        category = self.specialization
        product_pool = PRODUCTS_BY_CATEGORY.get(category, list(PRODUCTS_BY_CATEGORY.values())[0])

        products: List[Product] = []
        for _ in range(n):
            pname, pmin, pmax = random.choice(product_pool)
            original = round(random.uniform(pmin, pmax) * self.price_multiplier, 2)
            promo = random.choice(PROMOTIONS)
            discount = 0.20 if "20%" in promo else 0.0
            actual_price = round(original * (1 - discount), 2)
            qty = random.randint(0, 300)

            products.append(Product(
                name=pname,
                category=category,
                price=actual_price,
                original_price=original,
                quantity=qty,
                in_stock=(qty > 0),
                promotion=promo,
                rating=round(random.uniform(2.5, 5.0), 1),
                timestamp=datetime.now(),
                shop_id=self.shop_id,
            ))

        quality = max(0.1, min(1.0, self.base_quality + random.gauss(0, 0.05)))
        delay = max(1.0, random.expovariate(1 / (200 / self.network_stability)))

        return ShopDataBatch(
            shop_id=self.shop_id,
            shop_name=self.shop_name,
            specialization=self.specialization,
            emoji=self.emoji,
            products=products,
            timestamp=datetime.now(),
            data_quality=quality,
            transmission_delay_ms=delay,
        )


# ─── Фабрика генераторов ──────────────────────────────────────────────────────

class DataSourceManager:
    """Управляет всеми 10 генераторами магазинов."""

    def __init__(self, seed: int | None = None):
        self.generators = [ShopDataGenerator(i, seed) for i in range(len(SHOPS))]

    def fetch_all(self) -> List[ShopDataBatch]:
        return [g.generate_batch() for g in self.generators]

    def fetch_one(self, shop_id: int) -> ShopDataBatch:
        return self.generators[shop_id].generate_batch()

    def fetch_selected(self, shop_ids: List[int]) -> List[ShopDataBatch]:
        return [self.generators[i].generate_batch() for i in shop_ids]
