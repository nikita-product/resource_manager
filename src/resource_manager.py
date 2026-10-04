"""
Динамическое управление ресурсами для распределённой вычислительной системы.
"""

import time
import random
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Any


class TaskType(Enum):
    DATA_CLEANING     = "Очистка данных"
    ANALYTICS         = "Аналитика"
    AGGREGATION       = "Агрегация"
    ANOMALY_DETECTION = "Обнаружение аномалий"
    FORECASTING       = "Прогнозирование"


TASK_COLORS = {
    TaskType.DATA_CLEANING:     "#4ade80",
    TaskType.ANALYTICS:         "#60a5fa",
    TaskType.AGGREGATION:       "#f59e0b",
    TaskType.ANOMALY_DETECTION: "#f87171",
    TaskType.FORECASTING:       "#c084fc",
}


@dataclass
class TaskCost:
    cpu: float
    memory: float
    time_ms: float

    @property
    def total_weight(self) -> float:
        return self.cpu * 1.0 + self.memory * 0.01 + self.time_ms * 0.1


@dataclass
class ResourcePool:
    cpu: float
    memory: float
    time_ms: float

    def can_run(self, cost: TaskCost) -> bool:
        return (self.cpu >= cost.cpu and self.memory >= cost.memory and self.time_ms >= cost.time_ms)

    def allocate(self, cost: TaskCost):
        self.cpu -= cost.cpu; self.memory -= cost.memory; self.time_ms -= cost.time_ms

    def release_compute(self, cost: TaskCost):
        self.cpu += cost.cpu; self.memory += cost.memory

    def utilization(self, total: "ResourcePool") -> Dict[str, float]:
        return {
            "cpu":    1 - self.cpu / total.cpu    if total.cpu    > 0 else 0,
            "memory": 1 - self.memory / total.memory if total.memory > 0 else 0,
            "time":   1 - self.time_ms / total.time_ms if total.time_ms > 0 else 1,
        }

    def copy(self) -> "ResourcePool":
        return ResourcePool(self.cpu, self.memory, self.time_ms)


@dataclass
class Task:
    task_id: str
    shop_id: int
    shop_name: str
    task_type: TaskType
    cost: TaskCost
    expected_effect: float
    urgency: float = 1.0
    status: str = "pending"
    priority: float = 0.0
    result: Optional[Dict[str, Any]] = None
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    data_quality: float = 0.8
    delay_ms: float = 100.0

    def __lt__(self, other: "Task") -> bool:
        return self.priority > other.priority

    def recalculate_priority(self, pool: ResourcePool):
        weight = self.cost.total_weight
        efficiency = self.expected_effect / weight if weight > 0 else 0.0
        feasible   = 1.0 if pool.can_run(self.cost) else 0.1
        age_bonus  = math.log1p((time.time() - self.created_at) * 10) * 0.05
        self.priority = efficiency * feasible * self.urgency + age_bonus


# ─── Предсказатель выполнимости ───────────────────────────────────────────────

class FeasibilityPredictor:
    """
    Промежуточный алгоритм оценки вероятности успешного выполнения задачи
    ДО её постановки в очередь и расходования ресурсов.

    Оценка строится на взвешенной комбинации пяти независимых факторов:

      1. resource_margin  — запас ресурсов относительно стоимости задачи
                            (минимум из отношений CPU / памяти / времени).
                            Sigmoid-функция: margin=1 → P≈0.5, margin=2 → P≈0.88.

      2. data_quality     — качество входных данных от магазина (0–1).
                            Плохие данные увеличивают риск неполного выполнения.

      3. queue_pressure   — давление очереди: exp(-N/40).
                            При 0 задачах → 1.0, при 40 → ~0.37, при 100 → ~0.08.

      4. delay_penalty    — штраф за задержку сети: 1/(1 + delay/500).
                            delay=0мс → 1.0, delay=500мс → 0.5.

      5. complexity_score — сложность задачи: 1/(1 + weight/50).
                            Тяжёлые задачи несут больший риск нехватки ресурсов.

    Итоговая вероятность:
      P = σ(6 · (Σ wᵢfᵢ) − 3)

    Решение: P ≥ 0.70 → «Выполнить немедленно»
             P ≥ 0.45 → «Выполнить с мониторингом»
             P <  0.45 → «Отложить / Пересмотреть»
    """

    W_RESOURCE   = 0.40
    W_QUALITY    = 0.20
    W_QUEUE      = 0.15
    W_DELAY      = 0.15
    W_COMPLEXITY = 0.10

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    def predict(self, task: Task, pool: ResourcePool, queue_depth: int,
                data_quality: float = 0.8, delay_ms: float = 100.0) -> Dict[str, Any]:
        # Фактор 1: ресурсный запас
        cpu_margin  = pool.cpu    / task.cost.cpu    if task.cost.cpu    > 0 else 99.0
        mem_margin  = pool.memory / task.cost.memory if task.cost.memory > 0 else 99.0
        time_margin = pool.time_ms / task.cost.time_ms if task.cost.time_ms > 0 else 99.0
        bottleneck  = min(cpu_margin, mem_margin, time_margin)
        resource_score = self._sigmoid(2.5 * (bottleneck - 1.0))

        # Фактор 2: качество данных
        quality_score = float(data_quality)

        # Фактор 3: давление очереди
        queue_score = math.exp(-queue_depth / 40.0)

        # Фактор 4: задержка сети
        delay_score = 1.0 / (1.0 + delay_ms / 500.0)

        # Фактор 5: сложность
        complexity = task.cost.total_weight
        complexity_score = 1.0 / (1.0 + complexity / 50.0)

        weighted_sum = (
            self.W_RESOURCE   * resource_score  +
            self.W_QUALITY    * quality_score   +
            self.W_QUEUE      * queue_score      +
            self.W_DELAY      * delay_score      +
            self.W_COMPLEXITY * complexity_score
        )
        probability = self._sigmoid(6.0 * weighted_sum - 3.0)
        probability = round(max(0.01, min(0.99, probability)), 3)

        if probability >= 0.70:
            risk_level    = "🟢 Низкий"
            recommendation = "Выполнить немедленно"
        elif probability >= 0.45:
            risk_level    = "🟡 Средний"
            recommendation = "Выполнить с мониторингом"
        else:
            risk_level    = "🔴 Высокий"
            recommendation = "Отложить / Пересмотреть"

        margins = {"CPU": cpu_margin, "Память": mem_margin, "Время": time_margin}
        bottleneck_name = min(margins, key=margins.get)

        return {
            "probability":    probability,
            "risk_level":     risk_level,
            "recommendation": recommendation,
            "bottleneck":     bottleneck_name,
            "factors": {
                "Ресурсный запас":  round(resource_score, 3),
                "Качество данных":  round(quality_score, 3),
                "Давление очереди": round(queue_score, 3),
                "Задержка сети":    round(delay_score, 3),
                "Сложность задачи": round(complexity_score, 3),
            },
            "margins": {
                "CPU":    round(cpu_margin, 2),
                "Память": round(mem_margin, 2),
                "Время":  round(time_margin, 2),
            },
        }

    def predict_batch(self, tasks: List[Task], pool: ResourcePool,
                      queue_depth: int) -> List[Dict[str, Any]]:
        results = []
        for task in tasks:
            pred = self.predict(task, pool, queue_depth,
                                data_quality=task.data_quality, delay_ms=task.delay_ms)
            pred["task_id"]   = task.task_id
            pred["shop_name"] = task.shop_name
            pred["task_type"] = task.task_type.value
            pred["effect"]    = task.expected_effect
            pred["priority"]  = round(task.priority, 4)
            results.append(pred)
        return sorted(results, key=lambda r: r["probability"], reverse=True)


# ─── Менеджер ресурсов ────────────────────────────────────────────────────────

class ResourceManager:
    def __init__(self, total: ResourcePool):
        self.total      = total.copy()
        self.available  = total.copy()
        self._queue: List[Task] = []
        self.completed: List[Task] = []
        self.skipped:   List[Task] = []
        self.history:   List[Dict] = []
        self.total_effect = 0.0
        self._step = 0
        self.predictor = FeasibilityPredictor()

    def enqueue(self, task: Task):
        task.recalculate_priority(self.available)
        self._queue.append(task)
        self._sort_queue()

    def enqueue_many(self, tasks: List[Task]):
        for t in tasks:
            t.recalculate_priority(self.available)
        self._queue.extend(tasks)
        self._sort_queue()

    def _sort_queue(self):
        self._queue.sort(key=lambda t: t.priority, reverse=True)

    def _reprioritize(self):
        for t in self._queue:
            t.recalculate_priority(self.available)
        self._sort_queue()

    def step(self) -> Optional[Task]:
        if not self._queue:
            return None

        chosen: Optional[Task] = None
        for t in self._queue:
            if self.available.can_run(t.cost):
                chosen = t
                break

        if chosen is None:
            chosen = self._queue.pop(0)
            chosen.status = "skipped"
            self.skipped.append(chosen)
            self._reprioritize()
            return chosen

        self._queue.remove(chosen)
        self.available.allocate(chosen.cost)
        chosen.status = "running"
        chosen.result = self._simulate(chosen)
        chosen.status = "completed"
        chosen.completed_at = time.time()
        self.available.release_compute(chosen.cost)
        self.completed.append(chosen)
        self.total_effect += chosen.expected_effect
        self._step += 1
        self._reprioritize()
        self._log(chosen)
        return chosen

    def run(self, max_steps: int = 50) -> List[Task]:
        executed: List[Task] = []
        for _ in range(max_steps):
            if not self._queue:
                break
            t = self.step()
            if t:
                executed.append(t)
        return executed

    @staticmethod
    def _simulate(task: Task) -> Dict[str, Any]:
        r = random.Random(hash(task.task_id))
        if task.task_type == TaskType.DATA_CLEANING:
            return {"Записей обработано": r.randint(50, 2000), "Ошибок исправлено": r.randint(0, 120),
                    "Дубликатов удалено": r.randint(0, 50), "Качество данных +": f"{r.uniform(3, 25):.1f}%"}
        elif task.task_type == TaskType.ANALYTICS:
            return {"Средняя цена (₽)": round(r.uniform(200, 8000), 2),
                    "Тренд": r.choice(["▲ рост", "▼ снижение", "→ стабильно"]),
                    "Топ-категория": r.choice(["Электроника", "Одежда", "Продукты"]),
                    "Выручка (оценка ₽)": round(r.uniform(10_000, 1_000_000), 2),
                    "Конверсия": f"{r.uniform(1, 12):.1f}%"}
        elif task.task_type == TaskType.AGGREGATION:
            return {"Всего товаров": r.randint(100, 8000), "Категорий": r.randint(3, 10),
                    "Ср. кол-во (шт)": round(r.uniform(5, 250), 1),
                    "Общий объём (₽)": round(r.uniform(50_000, 5_000_000), 2),
                    "Уникальных SKU": r.randint(20, 500)}
        elif task.task_type == TaskType.ANOMALY_DETECTION:
            n = r.randint(0, 25)
            sev = "🟢 Низкая" if n < 5 else ("🟡 Средняя" if n < 15 else "🔴 Высокая")
            return {"Аномалий найдено": n, "Доля аномалий": f"{r.uniform(0, 15):.1f}%",
                    "Критичность": sev, "Затронуто товаров": r.randint(0, n * 3 + 1),
                    "Рекомендация": "Ручная проверка" if n > 10 else "Авто-исправление"}
        else:
            return {"Горизонт прогноза": "7 дней", "Изм. спроса": f"{r.uniform(-20, 35):+.1f}%",
                    "Уверенность": f"{r.uniform(55, 96):.0f}%",
                    "Рекоменд. действие": r.choice(["Увеличить запасы","Снизить цены","Запустить акцию","Ограничить закупки"]),
                    "Прогноз выручки (₽)": round(r.uniform(100_000, 3_000_000), 2)}

    def _log(self, task: Task):
        util = self.available.utilization(self.total)
        self.history.append({
            "шаг": self._step, "задача": task.task_id, "магазин": task.shop_name,
            "тип": task.task_type.value, "эффект": task.expected_effect,
            "∑эффект": self.total_effect, "приоритет": round(task.priority, 4),
            "CPU исп. %": round(util["cpu"] * 100, 1),
            "Память исп. %": round(util["memory"] * 100, 1),
            "Время ост. мс": round(self.available.time_ms, 1),
            "В очереди": len(self._queue), "статус": task.status,
        })

    def stats(self) -> Dict[str, Any]:
        util = self.available.utilization(self.total)
        return {
            "completed": len(self.completed), "skipped": len(self.skipped),
            "in_queue": len(self._queue), "total_effect": round(self.total_effect, 2),
            "cpu_util_pct": round(util["cpu"] * 100, 1),
            "memory_util_pct": round(util["memory"] * 100, 1),
            "time_consumed_ms": round(self.total.time_ms - self.available.time_ms, 1),
            "time_remaining_ms": round(self.available.time_ms, 1),
            "avail_cpu": round(self.available.cpu, 2),
            "avail_memory": round(self.available.memory, 2),
        }

    def queue_snapshot(self) -> List[Dict]:
        return [{"ID": t.task_id, "Магазин": t.shop_name, "Тип": t.task_type.value,
                 "Приоритет": round(t.priority, 4), "Эффект": round(t.expected_effect, 2),
                 "CPU": t.cost.cpu, "Память(МБ)": t.cost.memory, "Время(мс)": t.cost.time_ms,
                 "Выполнимо": "✅" if self.available.can_run(t.cost) else "❌"}
                for t in self._queue]

    def feasibility_report(self) -> List[Dict]:
        """Запускает предсказатель для всех задач в очереди."""
        return self.predictor.predict_batch(self._queue, self.available, len(self._queue))