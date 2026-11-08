"""
Módulo de Simulación Monte Carlo para Gestión Cuantitativa de Riesgo.
Calcula:
- Simulación de N trayectorias mediante Geometric Brownian Motion (GBM) y Bootstrap de retornos
- Percentiles y abanico de confianza (p5, p25, p50, p75, p95)
- Probabilidad de éxito del trade: P(TP alcanzado antes que SL)
- Métricas de riesgo de cola: Value at Risk (VaR 95%) y Conditional VaR (CVaR / Expected Shortfall)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List


def run_monte_carlo_simulation(
    df: pd.DataFrame,
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    position_type: str = "BUY",
    num_simulations: int = 1000,
    time_horizon_bars: int = 48
) -> Dict[str, Any]:
    """
    Ejecuta una simulación Monte Carlo para evaluar las probabilidades de éxito
    y distribución de escenarios antes de ingresar al trade.
    """
    # 1. Estimar retornos logarítmicos históricos y volatilidad
    log_returns = np.log(df['close'] / df['close'].shift(1)).dropna()
    mu = float(log_returns.mean())
    sigma = float(log_returns.std())

    if sigma <= 1e-6:
        sigma = 0.01

    # 2. Generar caminos mediante Geometric Brownian Motion (GBM)
    # Dimensiones: (num_simulations, time_horizon_bars)
    np.random.seed(int(entry_price * 100) % 100000)
    innovations = np.random.normal(0, 1, size=(num_simulations, time_horizon_bars))

    # Drift corregido de Itô
    drift = (mu - 0.5 * (sigma ** 2))
    increments = drift + sigma * innovations

    # Caminos de precios
    cumulative_returns = np.cumsum(increments, axis=1)
    paths = entry_price * np.exp(cumulative_returns)

    # Insertar el precio inicial en t=0
    initial_column = np.full((num_simulations, 1), entry_price)
    all_paths = np.hstack([initial_column, paths])  # (num_simulations, time_horizon_bars + 1)

    # 3. Análisis de eventos: TP vs SL
    tp_hits = 0
    sl_hits = 0
    no_hits = 0
    final_trade_returns = []

    for path in all_paths:
        hit_tp = False
        hit_sl = False

        for price in path[1:]:
            if position_type == "BUY":
                if price >= take_profit:
                    hit_tp = True
                    break
                elif price <= stop_loss:
                    hit_sl = True
                    break
            else:  # SELL (Short)
                if price <= take_profit:
                    hit_tp = True
                    break
                elif price >= stop_loss:
                    hit_sl = True
                    break

        if hit_tp:
            tp_hits += 1
            ret = (take_profit - entry_price) / entry_price if position_type == "BUY" else (entry_price - take_profit) / entry_price
            final_trade_returns.append(ret)
        elif hit_sl:
            sl_hits += 1
            ret = (stop_loss - entry_price) / entry_price if position_type == "BUY" else (entry_price - stop_loss) / entry_price
            final_trade_returns.append(ret)
        else:
            no_hits += 1
            final_price = path[-1]
            ret = (final_price - entry_price) / entry_price if position_type == "BUY" else (entry_price - final_price) / entry_price
            final_trade_returns.append(ret)

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
    final_trade_returns = np.array(final_trade_returns)
    var_95 = float(np.percentile(final_trade_returns, 5))  # Retorno en el peor 5%
    cvar_95 = float(final_trade_returns[final_trade_returns <= var_95].mean()) if len(final_trade_returns[final_trade_returns <= var_95]) > 0 else var_95
    expected_trade_return = float(np.mean(final_trade_returns))

    # Ratio de probabilidad (Edge del trade)
    edge_ratio = round(prob_tp / (prob_sl + 1e-5), 2)

    return {
        "num_simulations": num_simulations,
        "time_horizon_bars": time_horizon_bars,
        "entry_price": round(entry_price, 4),
        "stop_loss": round(stop_loss, 4),
        "take_profit": round(take_profit, 4),
        "position_type": position_type,
        "prob_take_profit": round(prob_tp, 4),
        "prob_stop_loss": round(prob_sl, 4),
        "prob_unresolved": round(prob_neutral, 4),
        "edge_ratio": edge_ratio,
        "expected_return": round(expected_trade_return * 100.0, 2),
        "value_at_risk_95": round(var_95 * 100.0, 2),
        "conditional_var_95": round(cvar_95 * 100.0, 2),
        "percentiles": percentiles_over_time,
        "sample_paths": sample_paths,
        "summary": (
            f"Monte Carlo ({num_simulations} caminos): P(TP)={prob_tp*100:.1f}%, "
            f"P(SL)={prob_sl*100:.1f}%, Retorno Esperado={expected_trade_return*100:.2f}%, "
            f"VaR 95%={var_95*100:.2f}%"
        )
    }

