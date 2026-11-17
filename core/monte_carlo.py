"""
Módulo de Simulación Monte Carlo para Gestión Cuantitativa de Riesgo.
Calcula:
- Simulación de N trayectorias mediante Geometric Brownian Motion (GBM) con drift histórico reducido
- Percentiles y abanico de confianza (p5, p25, p50, p75, p95)
- Probabilidad de éxito del trade: P(TP alcanzado antes que SL), incluyendo toques intra-vela
- Métricas de riesgo de cola: Value at Risk (VaR 95%) y Conditional VaR (CVaR / Expected Shortfall)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional


# Fracción del drift histórico que se conserva (0 = paseo aleatorio puro, 1 = extrapolar la tendencia completa)
DEFAULT_DRIFT_SHRINKAGE = 0.25
# Tope del drift por vela, en fracciones de sigma (evita que la tendencia domine sobre la volatilidad)
MAX_DRIFT_SIGMA_FRACTION = 0.05


def run_monte_carlo_simulation(
    df: pd.DataFrame,
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    position_type: str = "BUY",
    num_simulations: int = 1000,
    time_horizon_bars: int = 48,
    drift_shrinkage: float = DEFAULT_DRIFT_SHRINKAGE,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Ejecuta una simulación Monte Carlo para evaluar las probabilidades de éxito
    y distribución de escenarios antes de ingresar al trade.

    - Drift: la media de log-retornos ya incluye la corrección de Itô (mu - sigma²/2), así que se
      usa directamente (sin restar sigma²/2 otra vez). Se reduce con `drift_shrinkage` y se limita a
      MAX_DRIFT_SIGMA_FRACTION * sigma para no extrapolar la tendencia de los últimos días.
    - SL/TP: además del cierre de cada vela se detectan los toques intra-vela con la corrección de
      puente browniano (los precios pueden tocar el nivel y volver antes del cierre).
    - RNG local (no altera el generador aleatorio global de numpy).
    """
    position_type = "SELL" if str(position_type).upper() == "SELL" else "BUY"

    # 1. Estimar retornos logarítmicos históricos y volatilidad
    log_returns = np.log(df['close'] / df['close'].shift(1)).dropna()
    hist_log_drift = float(log_returns.mean()) if len(log_returns) else 0.0
    sigma = float(log_returns.std()) if len(log_returns) > 1 else 0.0

    if not np.isfinite(sigma) or sigma <= 1e-6:
        sigma = 0.01
    if not np.isfinite(hist_log_drift):
        hist_log_drift = 0.0

    shrink = float(min(max(drift_shrinkage, 0.0), 1.0))
    drift_cap = MAX_DRIFT_SIGMA_FRACTION * sigma
    drift = float(np.clip(hist_log_drift * shrink, -drift_cap, drift_cap))

    # 2. Generar caminos log-normales (GBM discretizado). Semilla local reproducible por precio.
    if seed is None:
        seed = int(entry_price * 100) % 100000
    rng = np.random.default_rng(seed)
    innovations = rng.standard_normal(size=(num_simulations, time_horizon_bars))
    increments = drift + sigma * innovations

    log_entry = np.log(entry_price)
    log_paths = np.hstack([
        np.full((num_simulations, 1), log_entry),
        log_entry + np.cumsum(increments, axis=1)
    ])  # (num_simulations, time_horizon_bars + 1)
    all_paths = np.exp(log_paths)

    # 3. Análisis de eventos: TP vs SL (cierre de vela + toque intra-vela por puente browniano)
    def _first_touch(level: float, upper: bool) -> np.ndarray:
        """Índice (1..H) de la primera vela que toca `level`, o H+1 si nunca lo toca."""
        log_level = np.log(level)
        x0, x1 = log_paths[:, :-1], log_paths[:, 1:]
        if upper:
            closed_beyond = x1 >= log_level
            d0, d1 = log_level - x0, log_level - x1
        else:
            closed_beyond = x1 <= log_level
            d0, d1 = x0 - log_level, x1 - log_level
        # P(el camino cruza el nivel dentro de la vela | extremos x0, x1 del mismo lado)
        with np.errstate(over="ignore", invalid="ignore"):
            p_cross = np.where((d0 > 0) & (d1 > 0), np.exp(-2.0 * d0 * d1 / (sigma ** 2)), 0.0)
        touched = closed_beyond | (rng.random(p_cross.shape) < p_cross)
        return np.where(touched.any(axis=1), touched.argmax(axis=1) + 1, time_horizon_bars + 1)

    if position_type == "BUY":
        tp_idx = _first_touch(take_profit, upper=True)
        sl_idx = _first_touch(stop_loss, upper=False)
    else:
        tp_idx = _first_touch(take_profit, upper=False)
        sl_idx = _first_touch(stop_loss, upper=True)

    never = time_horizon_bars + 1
    # Si ambos niveles se tocan en la misma vela no se sabe cuál fue primero -> se asume SL (conservador)
    hit_sl = (sl_idx < never) & (sl_idx <= tp_idx)
    hit_tp = (tp_idx < never) & ~hit_sl
    unresolved = ~(hit_tp | hit_sl)

    direction = 1.0 if position_type == "BUY" else -1.0
    tp_ret = direction * (take_profit - entry_price) / entry_price
    sl_ret = direction * (stop_loss - entry_price) / entry_price
    final_ret = direction * (all_paths[:, -1] - entry_price) / entry_price
    final_trade_returns = np.where(hit_tp, tp_ret, np.where(hit_sl, sl_ret, final_ret))

    tp_hits = int(hit_tp.sum())
    sl_hits = int(hit_sl.sum())
    no_hits = int(unresolved.sum())

    prob_tp = float(tp_hits / num_simulations)
    prob_sl = float(sl_hits / num_simulations)
    prob_neutral = float(no_hits / num_simulations)

    # 4. Cálculo de percentiles a lo largo del tiempo para visualización gráfica
    percentiles_over_time = {
        "p5": np.percentile(all_paths, 5, axis=0).tolist(),
        "p25": np.percentile(all_paths, 25, axis=0).tolist(),
        "p50": np.percentile(all_paths, 50, axis=0).tolist(),
        "p75": np.percentile(all_paths, 75, axis=0).tolist(),
        "p95": np.percentile(all_paths, 95, axis=0).tolist(),
    }

    # Muestra de 15 caminos representativos para renderizar en gráficos
    sample_indices = np.linspace(0, num_simulations - 1, 15, dtype=int)
    sample_paths = all_paths[sample_indices].tolist()

    # 5. Métricas de riesgo de cola
    var_95 = float(np.percentile(final_trade_returns, 5))  # Retorno en el peor 5%
    cvar_95 = float(final_trade_returns[final_trade_returns <= var_95].mean()) if len(final_trade_returns[final_trade_returns <= var_95]) > 0 else var_95
    expected_trade_return = float(np.mean(final_trade_returns))

    # Ratio de probabilidad (Edge del trade)
    edge_ratio = round(prob_tp / (prob_sl + 1e-5), 2)

    # Últimas 24 velas históricas reales para conectar historia (T-23..T=0) con el cono (T=0..T+H)
    hist_tail = [float(x) for x in df['close'].tail(24).tolist()] if len(df) > 0 else [float(entry_price)]
    if hist_tail:
        hist_tail[-1] = float(entry_price)

    return {
        "num_simulations": num_simulations,
        "time_horizon_bars": time_horizon_bars,
        "entry_price": round(entry_price, 4),
        "stop_loss": round(stop_loss, 4),
        "take_profit": round(take_profit, 4),
        "position_type": position_type,
        "sigma_per_bar": round(sigma, 6),
        "historical_drift_per_bar": round(hist_log_drift, 8),
        "drift_per_bar_used": round(drift, 8),
        "prob_take_profit": round(prob_tp, 4),
        "prob_stop_loss": round(prob_sl, 4),
        "prob_unresolved": round(prob_neutral, 4),
        "edge_ratio": edge_ratio,
        "expected_return": round(expected_trade_return * 100.0, 2),
        "value_at_risk_95": round(var_95 * 100.0, 2),
        "conditional_var_95": round(cvar_95 * 100.0, 2),
        "historical_prices": [round(x, 4) for x in hist_tail],
        "percentiles": percentiles_over_time,
        "sample_paths": sample_paths,
        "summary": (
            f"Monte Carlo ({num_simulations} caminos): P(TP)={prob_tp*100:.1f}%, "
            f"P(SL)={prob_sl*100:.1f}%, Retorno Esperado={expected_trade_return*100:.2f}%, "
            f"VaR 95%={var_95*100:.2f}%"
        )
    }


def run_monte_carlo_sensitivity(
    df: pd.DataFrame,
    entry_price: float,
    atr: float,
    position_type: str = "BUY",
    num_simulations: int = 400,
    time_horizon_bars: int = 48,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Ejecuta un Análisis de Sensibilidad y Stress Testing Bidimensional (5 x 4):
    - Filas: Shocks de Volatilidad (0.75x, 1.00x, 1.25x, 1.50x, 2.00x sigma)
    - Columnas: Configuraciones de Stop Loss / Take Profit en múltiplos de ATR
      ((1.0, 2.0), (1.5, 2.5), (2.0, 3.2), (2.5, 4.5))
    """
    position_type = "SELL" if str(position_type).upper() == "SELL" else "BUY"
    effective_atr = max(float(atr), entry_price * 0.005)

    log_returns = np.log(df['close'] / df['close'].shift(1)).dropna()
    hist_log_drift = float(log_returns.mean()) if len(log_returns) else 0.0
    base_sigma = float(log_returns.std()) if len(log_returns) > 1 else 0.01
    if not np.isfinite(base_sigma) or base_sigma <= 1e-6:
        base_sigma = 0.01
    if not np.isfinite(hist_log_drift):
        hist_log_drift = 0.0

    vol_scenarios = [
        {"mult": 0.75, "label": "0.75x σ (Calma)"},
        {"mult": 1.00, "label": "1.00x σ (Base Actual)"},
        {"mult": 1.25, "label": "1.25x σ (Moderada)"},
        {"mult": 1.50, "label": "1.50x σ (Shock +50%)"},
        {"mult": 2.00, "label": "2.00x σ (Cisne Negro)"},
    ]
    sltp_configs = [
        {"sl_mult": 1.0, "tp_mult": 2.0, "label": "SL 1.0x / TP 2.0x"},
        {"sl_mult": 1.5, "tp_mult": 2.5, "label": "SL 1.5x / TP 2.5x"},
        {"sl_mult": 2.0, "tp_mult": 3.2, "label": "SL 2.0x / TP 3.2x"},
        {"sl_mult": 2.5, "tp_mult": 4.5, "label": "SL 2.5x / TP 4.5x"},
    ]

    if seed is None:
        seed = int(entry_price * 100) % 100000
    rng = np.random.default_rng(seed)
    base_innovations = rng.standard_normal(size=(num_simulations, time_horizon_bars))
    uniform_draws = rng.random(size=(num_simulations, time_horizon_bars))
    log_entry = np.log(entry_price)
    direction = 1.0 if position_type == "BUY" else -1.0
    never = time_horizon_bars + 1

    rows: List[Dict[str, Any]] = []
    best_cell: Optional[Dict[str, Any]] = None

    for r_idx, v_sc in enumerate(vol_scenarios):
        sigma = base_sigma * v_sc["mult"]
        drift_cap = MAX_DRIFT_SIGMA_FRACTION * sigma
        drift = float(np.clip(hist_log_drift * DEFAULT_DRIFT_SHRINKAGE, -drift_cap, drift_cap))

        increments = drift + sigma * base_innovations
        log_paths = np.hstack([
            np.full((num_simulations, 1), log_entry),
            log_entry + np.cumsum(increments, axis=1)
        ])
        all_paths = np.exp(log_paths)
        x0, x1 = log_paths[:, :-1], log_paths[:, 1:]

        def _touch_idx(level: float, upper: bool) -> np.ndarray:
            log_lvl = np.log(max(level, 1e-8))
            if upper:
                closed_beyond = x1 >= log_lvl
                d0, d1 = log_lvl - x0, log_lvl - x1
            else:
                closed_beyond = x1 <= log_lvl
                d0, d1 = x0 - log_lvl, x1 - log_lvl
            with np.errstate(over="ignore", invalid="ignore"):
                p_cross = np.where((d0 > 0) & (d1 > 0), np.exp(-2.0 * d0 * d1 / (sigma ** 2)), 0.0)
            touched = closed_beyond | (uniform_draws < p_cross)
            return np.where(touched.any(axis=1), touched.argmax(axis=1) + 1, never)

        cells: List[Dict[str, Any]] = []
        for c_idx, cfg in enumerate(sltp_configs):
            sl_dist = effective_atr * cfg["sl_mult"]
            tp_dist = effective_atr * cfg["tp_mult"]
            if position_type == "BUY":
                sl_price = max(entry_price - sl_dist, entry_price * 0.1)
                tp_price = entry_price + tp_dist
                tp_idx = _touch_idx(tp_price, upper=True)
                sl_idx = _touch_idx(sl_price, upper=False)
            else:
                sl_price = entry_price + sl_dist
                tp_price = max(entry_price - tp_dist, entry_price * 0.1)
                tp_idx = _touch_idx(tp_price, upper=False)
                sl_idx = _touch_idx(sl_price, upper=True)

            hit_sl = (sl_idx < never) & (sl_idx <= tp_idx)
            hit_tp = (tp_idx < never) & ~hit_sl

            tp_ret = direction * (tp_price - entry_price) / entry_price
            sl_ret = direction * (sl_price - entry_price) / entry_price
            final_ret = direction * (all_paths[:, -1] - entry_price) / entry_price
            trade_rets = np.where(hit_tp, tp_ret, np.where(hit_sl, sl_ret, final_ret))

            p_tp = float(hit_tp.mean())
            p_sl = float(hit_sl.mean())
            exp_ret_pct = round(float(np.mean(trade_rets)) * 100.0, 2)
            var_95_pct = round(float(np.percentile(trade_rets, 5)) * 100.0, 2)
            edge = round(p_tp / (p_sl + 1e-5), 2)

            cell_data = {
                "row_idx": r_idx,
                "col_idx": c_idx,
                "vol_mult": v_sc["mult"],
                "sl_mult": cfg["sl_mult"],
                "tp_mult": cfg["tp_mult"],
                "config_label": cfg["label"],
                "prob_tp_pct": round(p_tp * 100.0, 1),
                "prob_sl_pct": round(p_sl * 100.0, 1),
                "expected_return_pct": exp_ret_pct,
                "var_95_pct": var_95_pct,
                "edge_ratio": edge,
            }
            cells.append(cell_data)

            # Evaluar óptimo en volatilidad base (1.0x)
            if v_sc["mult"] == 1.00:
                if best_cell is None or exp_ret_pct > best_cell["expected_return_pct"]:
                    best_cell = cell_data

        rows.append({
            "vol_mult": v_sc["mult"],
            "vol_label": v_sc["label"],
            "sigma_bar_pct": round(sigma * 100.0, 3),
            "cells": cells,
        })

    # Buscar cómo se comporta la columna óptima bajo shock de +50% (1.50x sigma)
    opt_col = best_cell["col_idx"] if best_cell else 2
    shock_cell = rows[3]["cells"][opt_col] if len(rows) > 3 else (best_cell or {})
    opt_label = best_cell["config_label"] if best_cell else "SL 2.0x / TP 3.2x"
    opt_er = best_cell["expected_return_pct"] if best_cell else 0.0
    opt_ptp = best_cell["prob_tp_pct"] if best_cell else 0.0
    shock_er = shock_cell.get("expected_return_pct", 0.0)

    conclusion = (
        f"Óptimo en régimen actual (1.0x σ): {opt_label} ATR con E[R]={opt_er:+.2f}% y P(TP)={opt_ptp:.1f}%. "
        f"Bajo shock de estrés (+50% volatilidad, 1.50x σ), dicho perfil registra E[R]={shock_er:+.2f}%."
    )

    return {
        "position_type": position_type,
        "columns": [c["label"] for c in sltp_configs],
        "rows": rows,
        "optimal": best_cell,
        "conclusion": conclusion,
    }


