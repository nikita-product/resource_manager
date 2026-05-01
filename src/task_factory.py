"""
Фабрика задач — преобразует пакеты данных от магазинов в вычислительные задачи.
Каждый пакет порождает несколько задач разного типа,
с автоматическим расчётом стоимости и ожидаемого эффекта.
"""

import random
import math
from typing import List

from src.data_generator import ShopDataBatch
from src.resource_manager import Task, TaskCost, TaskType


# ─── Параметры задач по типу ──────────────────────────────────────────────────

# (базовый_cpu, базовая_память_МБ, базовое_время_мс, базовый_эффект)
TASK_PARAMS = {
    TaskType.DATA_CLEANING:     (2.0,  64.0,  120.0,  8.0),
    TaskType.ANALYTICS:         (4.0, 128.0,  300.0, 15.0),
    TaskType.AGGREGATION:       (3.0,  96.0,  200.0, 12.0),
    TaskType.ANOMALY_DETECTION: (5.0, 192.0,  400.0, 20.0),
    TaskType.FORECASTING:       (6.0, 256.0,  600.0, 25.0),
}

_task_counter = 0


def _next_id(shop_id: int, ttype: TaskType) -> str:
    global _task_counter
    _task_counter += 1
    abbrev = {
        TaskType.DATA_CLEANING:     "CLN",
        TaskType.ANALYTICS:         "ANA",
        TaskType.AGGREGATION:       "AGG",
        TaskType.ANOMALY_DETECTION: "ANM",
        TaskType.FORECASTING:       "FRC",
    }[ttype]
    return f"S{shop_id}-{abbrev}-{_task_counter:04d}"


def _scale_cost(base_cpu, base_mem, base_time, batch: ShopDataBatch) -> TaskCost:
    """
    Масштабируем стоимость задачи в зависимости от объёма пакета.
    Чем больше товаров и ниже качество — тем дороже задача.
    """
    volume_factor  = math.log1p(batch.batch_size / 10) / math.log1p(5)  # ~1 при 40 записях
    quality_factor = 2.0 - batch.data_quality                            # хуже качество → дороже
    scale = volume_factor * quality_factor

    noise = lambda: random.uniform(0.85, 1.15)
    return TaskCost(
        cpu    = round(base_cpu    * scale * noise(), 2),
        memory = round(base_mem    * scale * noise(), 2),
        time_ms= round(base_time   * scale * noise(), 2),
    )


def _expected_effect(base_effect: float, batch: ShopDataBatch, ttype: TaskType) -> float:
    """
    Ожидаемый эффект зависит от:
      - базового значения по типу задачи;
      - объёма данных (больше данных → больше пользы);
      - качества данных;
      - стоимости товаров в пакете (для аналитических задач).
    """
    volume_bonus  = math.log1p(batch.batch_size) / math.log1p(50)
    quality_bonus = batch.data_quality
    value_bonus   = math.log1p(batch.total_value / 1_000_000) * 2 if batch.total_value > 0 else 0

    type_bonus = {
        TaskType.DATA_CLEANING:     quality_bonus * 0.5,
        TaskType.ANALYTICS:         value_bonus,
        TaskType.AGGREGATION:       volume_bonus * 0.8,
        TaskType.ANOMALY_DETECTION: (1 - quality_bonus) * 2,  # нужнее при плохих данных
        TaskType.FORECASTING:       value_bonus * 1.2,
    }[ttype]

    noise = random.uniform(0.9, 1.1)
    return round((base_effect + type_bonus) * volume_bonus * noise, 3)


def make_tasks_from_batch(batch: ShopDataBatch,
                          task_types: List[TaskType] | None = None,
                          urgency_override: float | None = None) -> List[Task]:
    """
    Создаёт список задач для одного пакета данных.

    :param batch:             пакет от магазина
    :param task_types:        список типов задач (None = все)
    :param urgency_override:  принудительный коэффициент срочности
    """
    if task_types is None:
        task_types = list(TaskType)

    urgency_base = 1.0 / (1.0 + batch.transmission_delay_ms / 500)  # задержка снижает срочность

    tasks: List[Task] = []
    for ttype in task_types:
        base_cpu, base_mem, base_time, base_eff = TASK_PARAMS[ttype]
        cost   = _scale_cost(base_cpu, base_mem, base_time, batch)
        effect = _expected_effect(base_eff, batch, ttype)
        urgency = urgency_override if urgency_override is not None else (
            urgency_base * random.uniform(0.8, 1.3)
        )

        tasks.append(Task(
            task_id=_next_id(batch.shop_id, ttype),
            shop_id=batch.shop_id,
            shop_name=f"{batch.emoji} {batch.shop_name}",
            task_type=ttype,
            cost=cost,
            expected_effect=effect,
            urgency=round(urgency, 3),
        ))

    return tasks


def make_tasks_from_batches(batches: List[ShopDataBatch],
                             task_types: List[TaskType] | None = None) -> List[Task]:
    """Создаёт задачи для нескольких пакетов сразу."""
    all_tasks: List[Task] = []
    for batch in batches:
        all_tasks.extend(make_tasks_from_batch(batch, task_types))
    return all_tasks
