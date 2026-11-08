"""
Módulo de Indicadores Técnicos y Matemáticos Cuantitativos.
Implementación pura y vectorizada en NumPy/Pandas para máxima portabilidad y velocidad.
Incluye:
- Medias Móviles de Regresión Lineal (Slope, Intercept, Curva de Regresión)
- MACD (Moving Average Convergence Divergence) + Histograma + Signal
- RSI (Relative Strength Index)
- Oscilador Estocástico (%K y %D)
- SAR Parabólico (Parabolic SAR)
- ATR (Average True Range) para Stop Loss / Take Profit dinámico
- Bandas de Bollinger y EMAs
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any


def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    """Calcula la Media Móvil Exponencial (EMA)."""
    return series.ewm(span=period, adjust=False).mean()


def calculate_sma(series: pd.Series, period: int) -> pd.Series:
    """Calcula la Media Móvil Simple (SMA)."""
    return series.rolling(window=period).mean()


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calcula el Average True Range (ATR) de Wilder.
    TR = max(High - Low, |High - Close_prev|, |Low - Close_prev|)
    """
    high = df['high']
    low = df['low']
    close = df['close']
    close_prev = close.shift(1)

    tr1 = high - low
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Wilder's Smoothing: ewm con alpha = 1 / period
    atr = tr.ewm(alpha=1.0 / period, adjust=False).mean()
    return atr


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Calcula el Relative Strength Index (RSI) con suavizado de Wilder.
    RSI = 100 - (100 / (1 + RS))
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()

    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def calculate_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal_period: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calcula el MACD:
    - MACD Line = EMA(fast) - EMA(slow)
    - Signal Line = EMA(MACD, signal_period)
    - Histogram = MACD - Signal
    """
    ema_fast = calculate_ema(series, fast)
    ema_slow = calculate_ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal_period)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calculate_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
    """
    Calcula el Oscilador Estocástico:
    %K = ((Close - Lowest Low) / (Highest High - Lowest Low)) * 100
    %D = SMA(%K, d_period)
    """
    low_min = df['low'].rolling(window=k_period).min()
    high_max = df['high'].rolling(window=k_period).max()
    denom = (high_max - low_min).replace(0, 1e-10)
    stoch_k = 100.0 * ((df['close'] - low_min) / denom)
    stoch_d = stoch_k.rolling(window=d_period).mean()
    return stoch_k, stoch_d


def calculate_linear_regression(series: pd.Series, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calcula la Media Móvil de Regresión Lineal:
    Ajusta una recta y = m * x + b en una ventana deslizante de tamaño 'period'.
    Retorna:
    - lr_curve: Último punto proyectado de la regresión y(period-1)
    - slope: Pendiente m de la recta (tendencia direccional)
    - r_squared: Coeficiente de determinación R^2 (fuerza de la tendencia)
    """
    x = np.arange(period)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()

    n = len(series)
    lr_curve = np.full(n, np.nan)
    slope = np.full(n, np.nan)
    r_squared = np.full(n, np.nan)

    values = series.to_numpy()

    for i in range(period - 1, n):
        y = values[i - period + 1 : i + 1]
        if np.isnan(y).any():
            continue
        y_mean = y.mean()
        cov = ((x - x_mean) * (y - y_mean)).sum()
        m = cov / x_var
        b = y_mean - m * x_mean
        proj = m * (period - 1) + b

        y_pred = m * x + b
        ss_tot = ((y - y_mean) ** 2).sum()
        ss_res = ((y - y_pred) ** 2).sum()
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-10 else 0.0

        lr_curve[i] = proj
        slope[i] = m
        r_squared[i] = max(0.0, min(1.0, r2))

    return (
        pd.Series(lr_curve, index=series.index),
        pd.Series(slope, index=series.index),
        pd.Series(r_squared, index=series.index)
    )


def calculate_parabolic_sar(df: pd.DataFrame, af_step: float = 0.02, af_max: float = 0.20) -> pd.Series:
    """
    Calcula el SAR Parabólico (Parabolic Stop and Reverse).
    Ideal para detección de puntos de reversión y trailing stops.
    """
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    n = len(df)

    sar = np.zeros(n)
    if n < 2:
        return pd.Series(sar, index=df.index)

    # Inicialización
    bull = close[1] >= close[0]
    af = af_step
    ep = high[0] if bull else low[0]
    sar[0] = low[0] if bull else high[0]

    for i in range(1, n):
        prev_sar = sar[i - 1]
        if bull:
            sar[i] = prev_sar + af * (ep - prev_sar)
            # Regla de restricción: SAR no debe ser mayor que los dos mínimos anteriores
            sar[i] = min(sar[i], low[i - 1], low[i - 2] if i > 1 else low[i - 1])

            if low[i] < sar[i]:
                # Reversión a tendencia bajista
                bull = False
                sar[i] = ep
                ep = low[i]
                af = af_step
            else:
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + af_step, af_max)
        else:
            sar[i] = prev_sar - af * (prev_sar - ep)
            # Regla de restricción: SAR no debe ser menor que los dos máximos anteriores
            sar[i] = max(sar[i], high[i - 1], high[i - 2] if i > 1 else high[i - 1])

            if high[i] > sar[i]:
                # Reversión a tendencia alcista
                bull = True
                sar[i] = ep
                ep = high[i]
                af = af_step
            else:
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + af_step, af_max)

    return pd.Series(sar, index=df.index)


def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula todos los indicadores técnicos y los concatena en el DataFrame.
    Requiere columnas: ['open', 'high', 'low', 'close', 'volume'].
    """
    res = df.copy()

    # EMAs
    res['ema_9'] = calculate_ema(res['close'], 9)
    res['ema_21'] = calculate_ema(res['close'], 21)
    res['ema_50'] = calculate_ema(res['close'], 50)
    res['ema_200'] = calculate_ema(res['close'], 200)

    # Volatilidad & Riesgo
    res['atr_14'] = calculate_atr(res, 14)

    # Momentum
    res['rsi_14'] = calculate_rsi(res['close'], 14)
    macd, macd_sig, macd_hist = calculate_macd(res['close'])
    res['macd'] = macd
    res['macd_signal'] = macd_sig
    res['macd_hist'] = macd_hist

    # Oscilador Estocástico
    stoch_k, stoch_d = calculate_stochastic(res)
    res['stoch_k'] = stoch_k
    res['stoch_d'] = stoch_d

    # Regresión Lineal
    lr_curve, slope, r2 = calculate_linear_regression(res['close'], 14)
    res['lr_curve'] = lr_curve
    res['lr_slope'] = slope
    res['lr_r2'] = r2

    # SAR Parabólico
    res['parabolic_sar'] = calculate_parabolic_sar(res)

    # Bandas de Bollinger (20, 2)
    sma_20 = calculate_sma(res['close'], 20)
    rolling_std = res['close'].rolling(20).std()
    res['bb_upper'] = sma_20 + (2.0 * rolling_std)
    res['bb_lower'] = sma_20 - (2.0 * rolling_std)
    res['bb_pct'] = (res['close'] - res['bb_lower']) / (res['bb_upper'] - res['bb_lower'] + 1e-10)

    # Log Returns
    res['log_return'] = np.log(res['close'] / res['close'].shift(1))

    return res

