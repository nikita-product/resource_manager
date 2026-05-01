"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Управление ресурсами в распределённых информационно-вычислительных системах ║
║  Streamlit-приложение                                                        ║
╚══════════════════════════════════════════════════════════════════════════════╝

Запуск:
    streamlit run app.py
"""

import random
import time
import math
from copy import deepcopy
from typing import List, Dict, Any

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.data_generator import DataSourceManager, ShopDataBatch, SHOPS
from src.resource_manager import (
    ResourceManager, ResourcePool, TaskCost, TaskType, Task, TASK_COLORS
)
from src.task_factory import make_tasks_from_batches, make_tasks_from_batch

# ═══════════════════════════════════════════════════════════════════════════════
# Конфигурация страницы
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="ДИВС — Управление ресурсами",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Кастомные стили
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Unbounded:wght@300;400;700&display=swap');

  html, body, [class*="css"] { font-family: 'JetBrains Mono', monospace; }
  h1, h2, h3 { font-family: 'Unbounded', sans-serif !important; letter-spacing: -0.5px; }

  .stMetric { background: #0f172a; border: 1px solid #1e3a5f; border-radius: 12px; padding: 16px; }
  .stMetric label { color: #64748b !important; font-size: 0.7rem !important; text-transform: uppercase; }
  .stMetric [data-testid="stMetricValue"] { color: #38bdf8 !important; font-size: 1.5rem !important; }
  .stMetric [data-testid="stMetricDelta"] { font-size: 0.75rem !important; }

  div[data-testid="stSidebarContent"] { background: #020817; }

  .task-card {
    background: #0f172a;
    border-left: 4px solid #38bdf8;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 6px 0;
    font-size: 0.82rem;
  }
  .tag {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 99px;
    font-size: 0.7rem;
    font-weight: 700;
    margin-right: 4px;
  }
  .header-bar {
    background: linear-gradient(90deg, #0f172a, #1e3a5f);
    border-radius: 12px;
    padding: 20px 28px;
    margin-bottom: 20px;
    border: 1px solid #1e3a5f;
  }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Состояние сессии
# ═══════════════════════════════════════════════════════════════════════════════

def init_state():
    if "initialized" not in st.session_state:
        st.session_state.initialized   = True
        st.session_state.batches       = []
        st.session_state.manager       = None
        st.session_state.history       = []
        st.session_state.run_count     = 0
        st.session_state.total_rounds  = 0
        st.session_state.all_products_df = pd.DataFrame()
        st.session_state.pred_results  = []

init_state()

# ═══════════════════════════════════════════════════════════════════════════════
# Sidebar — конфигурация
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ⚙️ Конфигурация")
    st.divider()

    st.markdown("### 🗄 Ресурсы системы")
    total_cpu    = st.slider("CPU (ядра)",         4,  64, 16)
    total_memory = st.slider("Память (ГБ)",        4, 128, 32)
    total_time   = st.slider("Временной бюджет (сек)", 5, 120, 30)

    st.divider()
    st.markdown("### 🏪 Источники данных")
    active_shops = st.multiselect(
        "Активные магазины",
        options=[f"{s['emoji']} {s['name']}" for s in SHOPS],
        default=[f"{s['emoji']} {s['name']}" for s in SHOPS],
    )
    active_ids = [i for i, s in enumerate(SHOPS) if f"{s['emoji']} {s['name']}" in active_shops]

    st.divider()
    st.markdown("### 🔬 Типы задач")
    task_type_labels = {t.value: t for t in TaskType}
    selected_task_labels = st.multiselect(
        "Разрешённые типы",
        options=list(task_type_labels.keys()),
        default=list(task_type_labels.keys()),
    )
    selected_task_types = [task_type_labels[l] for l in selected_task_labels]

    st.divider()
    st.markdown("### ⚡ Режим выполнения")
    # ИСПРАВЛЕНИЕ: Значение по умолчанию снижено с 50 до 10, чтобы очередь не исчерпывалась моментально
    max_steps   = st.slider("Авто-выполнение шагов при старте", 0, 200, 10)
    seed        = st.number_input("Seed (0 = случайно)", 0, 9999, 42)
    use_seed    = seed > 0

    st.divider()
    btn_run  = st.button("▶ Запустить симуляцию",  type="primary", use_container_width=True)
    btn_add  = st.button("➕ Добавить раунд данных", use_container_width=True)
    btn_reset = st.button("🔄 Сброс", use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Вспомогательные функции
# ═══════════════════════════════════════════════════════════════════════════════

def build_manager(batches: List[ShopDataBatch]) -> ResourceManager:
    pool = ResourcePool(
        cpu    = float(total_cpu),
        memory = float(total_memory * 1024),       # ГБ → МБ
        time_ms= float(total_time  * 1000),        # сек → мс
    )
    mgr = ResourceManager(pool)
    tasks = make_tasks_from_batches(batches, selected_task_types or None)
    mgr.enqueue_many(tasks)
    return mgr

def collect_products_df(batches: List[ShopDataBatch]) -> pd.DataFrame:
    rows = []
    for b in batches:
        for p in b.products:
            rows.append({
                "Магазин":       f"{b.emoji} {b.shop_name}",
                "Специализация": b.specialization,
                "Товар":         p.name,
                "Категория":     p.category,
                "Цена (₽)":      p.price,
                "Цена до скидки (₽)": p.original_price,
                "Скидка %":      p.discount_pct,
                "Кол-во":        p.quantity,
                "В наличии":     "✅" if p.in_stock else "❌",
                "Акция":         p.promotion,
                "Рейтинг":       p.rating,
                "Оборот (₽)":    p.total_value,
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame()

def plotly_theme() -> Dict:
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor ="#0f172a",
        font=dict(color="#94a3b8", family="JetBrains Mono"),
        xaxis=dict(gridcolor="#1e293b", zerolinecolor="#1e293b"),
        yaxis=dict(gridcolor="#1e293b", zerolinecolor="#1e293b"),
    )

# ═══════════════════════════════════════════════════════════════════════════════
# Обработка кнопок
# ═══════════════════════════════════════════════════════════════════════════════

if btn_reset:
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

dsm = DataSourceManager(seed=seed if use_seed else None)

if btn_run:
    if not active_ids:
        st.error("Выберите хотя бы один магазин.")
    elif not selected_task_types:
        st.error("Выберите хотя бы один тип задачи.")
    else:
        with st.spinner("Генерация данных и запуск алгоритма..."):
            batches = dsm.fetch_selected(active_ids)
            mgr = build_manager(batches)
            mgr.run(max_steps=max_steps)
            
            st.session_state.batches       = batches
            st.session_state.manager       = mgr
            st.session_state.history       = deepcopy(mgr.history)
            st.session_state.run_count     = 1
            st.session_state.total_rounds  = 1
            st.session_state.all_products_df = collect_products_df(batches)
            st.session_state.pred_results  = mgr.feasibility_report() # Обновляем предсказания
        st.success("Симуляция запущена!")

if btn_add and st.session_state.manager is not None:
    with st.spinner("Добавление нового раунда данных..."):
        new_batches = dsm.fetch_selected(active_ids)
        mgr = st.session_state.manager
        new_tasks = make_tasks_from_batches(new_batches, selected_task_types or None)
        mgr.enqueue_many(new_tasks)
        mgr.run(max_steps=max_steps)
        
        st.session_state.batches.extend(new_batches)
        st.session_state.history = deepcopy(mgr.history)
        st.session_state.total_rounds += 1
        st.session_state.pred_results = mgr.feasibility_report()
        new_df = collect_products_df(new_batches)
        st.session_state.all_products_df = pd.concat(
            [st.session_state.all_products_df, new_df], ignore_index=True
        )
    st.success(f"Раунд {st.session_state.total_rounds} добавлен!")

# ═══════════════════════════════════════════════════════════════════════════════
# Заголовок
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="header-bar">
  <h1 style="margin:0;color:#38bdf8;font-size:1.5rem;">
    ⚙️ Управление ресурсами в ДИВС
  </h1>
  <p style="margin:4px 0 0 0;color:#64748b;font-size:0.8rem;">
    Динамическое распределение вычислительных ресурсов между задачами
    обработки данных от 10 торговых источников
  </p>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Вкладки
# ═══════════════════════════════════════════════════════════════════════════════

# Все вкладки сохранены, добавлена новая вкладка "Предсказание"
tabs = st.tabs([
    "📊 Дашборд",
    "🏪 Данные магазинов",
    "⚙️ Алгоритм и очередь",
    "📈 История выполнения",
    "🤖 Предсказание",
    "🗺 Описание модели",
    "📋 Сырые данные",
])

mgr: ResourceManager | None = st.session_state.manager
batches: List[ShopDataBatch] = st.session_state.batches
hist: List[Dict] = st.session_state.history

# ────────────────────────────────────────────────────────────────────────────
# ВК 1 — ДАШБОРД
# ────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    if mgr is None:
        st.info("▶ Нажмите **Запустить симуляцию** в боковой панели.")
    else:
        s = mgr.stats()

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("✅ Выполнено задач",  s["completed"])
        c2.metric("⏭ Пропущено",         s["skipped"])
        c3.metric("🎯 Суммарный эффект",  f"{s['total_effect']:.1f}")
        c4.metric("🖥 CPU использовано",  f"{s['cpu_util_pct']:.0f}%")
        c5.metric("💾 Память использована", f"{s['memory_util_pct']:.0f}%")
        c6.metric("⏱ Время осталось",    f"{s['time_remaining_ms']/1000:.1f}с")

        st.divider()
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("#### 📈 Накопленный эффект")
            if hist:
                df_h = pd.DataFrame(hist)
                fig = px.area(df_h, x="шаг", y="∑эффект",
                              color_discrete_sequence=["#38bdf8"],
                              template="plotly_dark")
                fig.update_layout(**plotly_theme(), height=300, margin=dict(t=20, b=20))
                st.plotly_chart(fig, use_container_width=True)

        with col_b:
            st.markdown("#### 🔋 Текущее использование ресурсов")
            fig2 = go.Figure()
            resources = [
                ("CPU",    s["cpu_util_pct"],    "#38bdf8"),
                ("Память", s["memory_util_pct"], "#a78bfa"),
                ("Время",  (s["time_consumed_ms"] / (total_time * 1000)) * 100, "#fb923c"),
            ]
            for name, val, color in resources:
                fig2.add_trace(go.Bar(
                    name=name, x=[name], y=[val],
                    marker_color=color, text=[f"{val:.1f}%"], textposition="outside"
                ))
            fig2.add_hline(y=100, line_dash="dash", line_color="#ef4444",
                           annotation_text="Лимит 100%")
            fig2.update_layout(**plotly_theme(), height=300,
                               yaxis_range=[0, 120], showlegend=False, margin=dict(t=20, b=20))
            st.plotly_chart(fig2, use_container_width=True)

        col_c, col_d = st.columns(2)
        with col_c:
            st.markdown("#### 🧩 Задачи по типам")
            if hist:
                df_types = pd.DataFrame(hist)["тип"].value_counts().reset_index()
                df_types.columns = ["тип", "кол-во"]
                colors_list = [TASK_COLORS.get(t, "#64748b") for t in [tt for tt in TaskType if tt.value in df_types["тип"].values]]
                fig3 = px.pie(df_types, names="тип", values="кол-во",
                              color_discrete_sequence=list(TASK_COLORS.values()),
                              template="plotly_dark", hole=0.4)
                fig3.update_layout(**plotly_theme(), height=300, margin=dict(t=20, b=20))
                st.plotly_chart(fig3, use_container_width=True)

        with col_d:
            st.markdown("#### 🏪 Эффект по магазинам")
            if hist:
                df_shops = pd.DataFrame(hist).groupby("магазин")["эффект"].sum().reset_index()
                df_shops = df_shops.sort_values("эффект", ascending=True)
                fig4 = px.bar(df_shops, x="эффект", y="магазин",
                              orientation="h", color="эффект", color_continuous_scale="Blues",
                              template="plotly_dark")
                fig4.update_layout(**plotly_theme(), height=300, coloraxis_showscale=False, margin=dict(t=20, b=20))
                st.plotly_chart(fig4, use_container_width=True)

        st.markdown("#### 🕐 Последние выполненные задачи")
        if mgr.completed:
            for task in reversed(mgr.completed[-5:]):
                color = TASK_COLORS.get(task.task_type, "#64748b")
                result_str = " | ".join(f"<b>{k}</b>: {v}" for k, v in (task.result or {}).items())
                st.markdown(f"""
                <div class="task-card" style="border-color:{color}">
                  <span class="tag" style="background:{color}22;color:{color};">
                    {task.task_type.value}
                  </span>
                  <b>{task.task_id}</b> &nbsp;·&nbsp; {task.shop_name}
                  &nbsp;·&nbsp; Эффект: <b style="color:#4ade80">{task.expected_effect:.2f}</b>
                  &nbsp;·&nbsp; Приоритет: <b>{task.priority:.3f}</b>
                  <br><small style="color:#64748b">{result_str}</small>
                </div>
                """, unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────────
# ВК 2 — ДАННЫЕ МАГАЗИНОВ
# ────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    if not batches:
        st.info("▶ Сначала запустите симуляцию.")
    else:
        st.markdown("### 🏪 Сводная таблица по магазинам")
        summary_rows = [b.to_stats_dict() for b in batches[-10:]]
        df_summary = pd.DataFrame(summary_rows)
        st.dataframe(df_summary, use_container_width=True, height=300)

        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 📦 Объём партий по магазинам")
            sizes = [{"Магазин": b.shop_name, "Товаров": b.batch_size, "Качество %": round(b.data_quality*100,1)} for b in batches]
            df_sz = pd.DataFrame(sizes).groupby("Магазин", as_index=False).sum()
            fig = px.bar(df_sz, x="Магазин", y="Товаров", color="Товаров", color_continuous_scale="Teal", template="plotly_dark")
            fig.update_layout(**plotly_theme(), height=320, margin=dict(t=20,b=70), xaxis_tickangle=-30, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### 📡 Задержка передачи данных")
            delays = [{"Магазин": b.shop_name, "Задержка (мс)": b.transmission_delay_ms} for b in batches]
            df_d = pd.DataFrame(delays).groupby("Магазин", as_index=False).mean()
            fig2 = px.bar(df_d, x="Магазин", y="Задержка (мс)", color="Задержка (мс)", color_continuous_scale="Reds", template="plotly_dark")
            fig2.update_layout(**plotly_theme(), height=320, margin=dict(t=20,b=70), xaxis_tickangle=-30, coloraxis_showscale=False)
            st.plotly_chart(fig2, use_container_width=True)

        df_prod = st.session_state.all_products_df
        if not df_prod.empty:
            st.markdown("#### 💰 Распределение цен (log-scale)")
            selected_shop = st.selectbox("Выберите магазин", ["Все"] + sorted(df_prod["Магазин"].unique().tolist()))
            df_filt = df_prod if selected_shop == "Все" else df_prod[df_prod["Магазин"]==selected_shop]
            fig3 = px.histogram(df_filt, x="Цена (₽)", color="Категория", nbins=50, log_y=True,
                                template="plotly_dark", color_discrete_sequence=px.colors.qualitative.Set3)
            fig3.update_layout(**plotly_theme(), height=350, margin=dict(t=20,b=40))
            st.plotly_chart(fig3, use_container_width=True)

            st.markdown("#### 🏆 Топ-10 товаров по обороту")
            top = df_filt.nlargest(10, "Оборот (₽)")[["Магазин","Товар","Категория","Цена (₽)","Кол-во","Оборот (₽)","Акция","Рейтинг"]]
            st.dataframe(top, use_container_width=True)

# ────────────────────────────────────────────────────────────────────────────
# ВК 3 — АЛГОРИТМ И ОЧЕРЕДЬ
# ────────────────────────────────────────────────────────────────────────────
with tabs[2]:
    if mgr is None:
        st.info("▶ Сначала запустите симуляцию.")
    else:
        col_info, col_queue = st.columns([1, 2])
        with col_info:
            s = mgr.stats()
            st.markdown("### 🔢 Текущее состояние")
            st.metric("Задач в очереди",   s["in_queue"])
            st.metric("Выполнено",         s["completed"])
            st.metric("Пропущено",         s["skipped"])
            st.metric("CPU свободно",      f"{s['avail_cpu']:.1f}")
            st.metric("Память свободно",   f"{s['avail_memory']:.0f} МБ")
            st.metric("Время осталось",    f"{s['time_remaining_ms']/1000:.2f} с")

            st.divider()
            st.markdown("#### ⚡ Шаг алгоритма")
            if st.button("▶ Выполнить 1 шаг", use_container_width=True):
                result = mgr.step()
                if result:
                    st.success(f"Выполнено: {result.task_id} ({result.status})")
                    st.session_state.history = deepcopy(mgr.history)
                    st.session_state.pred_results = mgr.feasibility_report()
                else:
                    st.warning("Нет выполнимых задач.")

            if st.button("▶▶ Выполнить 5 шагов", use_container_width=True):
                done = mgr.run(5)
                st.success(f"Выполнено {len(done)} задач.")
                st.session_state.history = deepcopy(mgr.history)
                st.session_state.pred_results = mgr.feasibility_report()

        with col_queue:
            st.markdown("### 📋 Очередь задач (по приоритету)")
            q_snap = mgr.queue_snapshot()
            if q_snap:
                df_q = pd.DataFrame(q_snap)
                st.dataframe(df_q, use_container_width=True, height=420)
                fig_s = px.scatter(df_q, x="Эффект", y="Приоритет", color="Тип", size="CPU",
                                   hover_data=["ID","Магазин"], template="plotly_dark",
                                   color_discrete_sequence=list(TASK_COLORS.values()))
                fig_s.update_layout(**plotly_theme(), height=350, title="Приоритет vs Ожидаемый эффект")
                st.plotly_chart(fig_s, use_container_width=True)
            else:
                st.success("Очередь пуста — все задачи обработаны!")

# ────────────────────────────────────────────────────────────────────────────
# ВК 4 — ИСТОРИЯ ВЫПОЛНЕНИЯ
# ────────────────────────────────────────────────────────────────────────────
with tabs[3]:
    if not hist:
        st.info("▶ Сначала запустите симуляцию.")
    else:
        df_hist = pd.DataFrame(hist)
        st.markdown("### 📈 Динамика выполнения")
        c1, c2 = st.columns(2)
        with c1:
            fig = px.line(df_hist, x="шаг", y=["CPU исп. %", "Память исп. %"], template="plotly_dark",
                          color_discrete_sequence=["#38bdf8", "#a78bfa"], labels={"value": "Использование %", "variable": ""})
            fig.add_hline(y=80, line_dash="dot", line_color="#f59e0b", annotation_text="Порог 80%", annotation_position="bottom right")
            fig.update_layout(**plotly_theme(), height=320, title="Нагрузка ресурсов по шагам")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig2 = px.bar(df_hist, x="шаг", y="эффект", color="тип", template="plotly_dark",
                          color_discrete_map={t.value: c for t, c in TASK_COLORS.items()}, labels={"эффект": "Эффект задачи"})
            fig2.update_layout(**plotly_theme(), height=320, title="Эффект каждого шага")
            st.plotly_chart(fig2, use_container_width=True)

        fig3 = px.scatter(df_hist, x="шаг", y="приоритет", color="тип", size="эффект", template="plotly_dark",
                          color_discrete_map={t.value: c for t, c in TASK_COLORS.items()})
        fig3.update_layout(**plotly_theme(), height=350, title="Приоритеты задач по шагам")
        st.plotly_chart(fig3, use_container_width=True)

        st.markdown("#### 📄 Журнал выполнения")
        st.dataframe(df_hist, use_container_width=True, height=400)

        st.markdown("#### 🗺 Тепловая карта: магазин × тип задачи")
        pivot = df_hist.pivot_table(values="эффект", index="магазин", columns="тип", aggfunc="sum", fill_value=0)
        fig_h = px.imshow(pivot, color_continuous_scale="Blues", template="plotly_dark", aspect="auto")
        fig_h.update_layout(**plotly_theme(), height=400)
        st.plotly_chart(fig_h, use_container_width=True)

# ────────────────────────────────────────────────────────────────────────────
# ВК 5 — ПРЕДСКАЗАНИЕ ВЫПОЛНИМОСТИ (ИНТЕГРАЦИЯ НОВОГО ФУНКЦИОНАЛА)
# ────────────────────────────────────────────────────────────────────────────
with tabs[4]:
    pred_results = st.session_state.get("pred_results", [])

    if mgr is None:
        st.info("▶ Сначала запустите симуляцию.")
    elif not pred_results:
        st.warning("Очередь задач пуста! Добавьте новый раунд данных или уменьшите количество шагов алгоритма для анализа предсказаний.")
    else:
        st.markdown("### 🤖 Оценка выполнимости задач в очереди")
        st.markdown(
            "Промежуточный алгоритм оценки вероятности успешного выполнения задачи ДО её постановки на ресурсы. "
            "Анализ основан на 5 метриках (ресурсный запас, качество данных, длина очереди, задержка сети, сложность)."
        )
        st.divider()

        total_preds = len(pred_results)
        high   = sum(1 for r in pred_results if r["probability"] >= 0.70)
        medium = sum(1 for r in pred_results if 0.45 <= r["probability"] < 0.70)
        low    = sum(1 for r in pred_results if r["probability"] < 0.45)
        avg_p  = sum(r["probability"] for r in pred_results) / total_preds if total_preds else 0

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Задач в анализе", total_preds)
        c2.metric("🟢 Низкий риск", high,   delta=f"{high/total_preds:.0%}" if total_preds else "—")
        c3.metric("🟡 Средний риск",  medium, delta=f"{medium/total_preds:.0%}" if total_preds else "—")
        c4.metric("🔴 Высокий риск",   low,    delta=f"{low/total_preds:.0%}" if total_preds else "—")
        c5.metric("Средняя P(успех)", f"{avg_p:.0%}")

        st.divider()
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("#### 📊 Распределение вероятностей")
            probs = [r["probability"] for r in pred_results]
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=probs, nbinsx=20, marker_color="#38bdf8", opacity=0.8, name="Задачи"
            ))
            fig_hist.add_vline(x=0.70, line_dash="dash", line_color="#4ade80", annotation_text="🟢 >0.70", annotation_position="top left")
            fig_hist.add_vline(x=0.45, line_dash="dash", line_color="#f59e0b", annotation_text="🟡 >0.45", annotation_position="top right")
            fig_hist.update_layout(**plotly_theme(), height=320, xaxis_title="P(успех)", yaxis_title="Кол-во задач", margin=dict(t=20, b=40))
            st.plotly_chart(fig_hist, use_container_width=True)

        with col_b:
            st.markdown("#### 🔍 Как признаки влияют на P(успех)")
            scatter_rows = [
                {
                    "Сложность задачи": r["factors"]["Сложность задачи"],
                    "P(успех)": r["probability"],
                    "Тип": r["task_type"],
                    "Рекомендация": r["recommendation"],
                    "Магазин": r["shop_name"],
                }
                for r in pred_results
            ]
            df_sc = pd.DataFrame(scatter_rows)
            fig_sc = px.scatter(
                df_sc, x="Сложность задачи", y="P(успех)", color="Тип",
                hover_data=["Магазин", "Рекомендация"], template="plotly_dark",
                color_discrete_sequence=list(TASK_COLORS.values()),
                trendline="ols" if len(df_sc) > 3 else None,
            )
            fig_sc.add_hline(y=0.70, line_dash="dash", line_color="#4ade80")
            fig_sc.add_hline(y=0.45, line_dash="dash", line_color="#f59e0b")
            fig_sc.update_layout(**plotly_theme(), height=320, margin=dict(t=20, b=20))
            st.plotly_chart(fig_sc, use_container_width=True)

        st.markdown("#### 📋 Детали предсказаний")
        rows = []
        for r in pred_results:
            rows.append({
                "ID задачи":       r["task_id"],
                "Магазин":         r["shop_name"],
                "Тип задачи":      r["task_type"],
                "P(успех)":        f'{r["probability"]:.0%}',
                "Уровень Риска":   r["risk_level"],
                "Рекомендация":    r["recommendation"],
                "Узкое место":     r["bottleneck"],
                "Кач. данных":     f'{r["factors"]["Качество данных"]:.0%}',
                "Сложность":       f'{r["factors"]["Сложность задачи"]:.2f}',
            })
        df_preds = pd.DataFrame(rows)
        st.dataframe(df_preds, use_container_width=True, height=380)

# ────────────────────────────────────────────────────────────────────────────
# ВК 6 — ОПИСАНИЕ МОДЕЛИ
# ────────────────────────────────────────────────────────────────────────────
with tabs[5]:
    st.markdown("""
## 🗺 Описание модели управления ресурсами

### Постановка задачи
Дан набор вычислительных задач, каждая из которых характеризуется:
- **Ожидаемым эффектом** — ценностью результата (нормированная метрика 0–∞)
- **Стоимостью** — потреблением трёх ресурсов: CPU, память, время

Требуется максимизировать суммарный эффект при ресурсных ограничениях.

---

### Ресурсная модель

| Ресурс | Тип | Возобновляемость |
|--------|-----|-----------------|
| CPU (ядра) | Вычислительный | ✅ Возобновляемый |
| Память (МБ) | Вычислительный | ✅ Возобновляемый |
| Время (мс) | Временной | ❌ Невозобновляемый |

После завершения задачи CPU и память возвращаются в пул.  
Время расходуется безвозвратно.

---

### Алгоритм приоритизации
priority(t) = efficiency(t) × feasibility(t) × urgency(t) + age_bonus(t)

где:
efficiency(t) = expected_effect(t) / total_cost_weight(t)

total_cost_weight(t) = cpu(t) × 1.0
+ memory(t) × 0.01
+ time_ms(t) × 0.1

feasibility(t) = 1.0  если ресурсов достаточно
= 0.1  если нет (задача не теряется)

urgency(t) = 1.0 / (1 + delay_ms / 500)  × ξ   # ξ ∈ [0.8, 1.3]

age_bonus(t) = ln(1 + age_sec × 10) × 0.05
### Типы задач и их параметры
| Тип | CPU | Память | Время | Базовый эффект |
|-----|-----|--------|-------|----------------|
| Очистка данных | 2 | 64 МБ | 120 мс | 8.0 |
| Аналитика | 4 | 128 МБ | 300 мс | 15.0 |
| Агрегация | 3 | 96 МБ | 200 мс | 12.0 |
| Обнаружение аномалий | 5 | 192 МБ | 400 мс | 20.0 |
| Прогнозирование | 6 | 256 МБ | 600 мс | 25.0 |

---

### Источники данных (10 магазинов)
""")

    cols = st.columns(5)
    for i, s in enumerate(SHOPS):
        with cols[i % 5]:
            st.markdown(f"""
            <div class="task-card">
              <b style="font-size:1.3rem">{s['emoji']}</b><br>
              <b>{s['name']}</b><br>
              <small style="color:#64748b">{s['specialization']}</small>
            </div>
            """, unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────────
# ВК 7 — СЫРЫЕ ДАННЫЕ
# ────────────────────────────────────────────────────────────────────────────
with tabs[6]:
    if st.session_state.all_products_df.empty:
        st.info("▶ Сначала запустите симуляцию.")
    else:
        df_prod = st.session_state.all_products_df

        st.markdown(f"### 📋 Все товары — {len(df_prod):,} записей")

        c1, c2, c3 = st.columns(3)
        with c1:
            shop_filter = st.multiselect("Магазин", df_prod["Магазин"].unique().tolist())
        with c2:
            cat_filter  = st.multiselect("Категория", df_prod["Категория"].unique().tolist())
        with c3:
            promo_filter = st.multiselect("Акция", df_prod["Акция"].unique().tolist())

        df_f = df_prod.copy()
        if shop_filter: df_f = df_f[df_f["Магазин"].isin(shop_filter)]
        if cat_filter:  df_f = df_f[df_f["Категория"].isin(cat_filter)]
        if promo_filter: df_f = df_f[df_f["Акция"].isin(promo_filter)]

        price_range = st.slider("Диапазон цен (₽)",
                                float(df_prod["Цена (₽)"].min()),
                                float(df_prod["Цена (₽)"].max()),
                                (float(df_prod["Цена (₽)"].min()),
                                 float(df_prod["Цена (₽)"].max())))
        df_f = df_f[(df_f["Цена (₽)"] >= price_range[0]) & (df_f["Цена (₽)"] <= price_range[1])]

        st.markdown(f"**Отфильтровано: {len(df_f):,} записей**")
        st.dataframe(df_f, use_container_width=True, height=500)

        st.markdown("#### 📊 Описательная статистика")
        st.dataframe(
            df_f[["Цена (₽)", "Кол-во", "Оборот (₽)", "Скидка %", "Рейтинг"]].describe().round(2),
            use_container_width=True
        )

        csv_bytes = df_f.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "⬇️ Скачать как CSV",
            data=csv_bytes,
            file_name="shop_data_export.csv",
            mime="text/csv",
        )