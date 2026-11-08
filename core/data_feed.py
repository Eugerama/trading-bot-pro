"""
Módulo de Ingesta y Normalización de Datos Financieros.
Soporta:
- Yahoo Finance (yfinance)
- Soporte para Tickers Cripto y Acciones
- Mecanismo de contingencia / Fail-safe offline con Movimiento Browniano Geométrico realista
"""

import os
import time
import numpy as np
import pandas as pd
from typing import Optional


# Mapeo de tickers comunes para yfinance
TICKER_MAP = {
    "BTCUSDT": "BTC-USD",
    "ETHUSDT": "ETH-USD",
    "SOLUSDT": "SOL-USD",
    "BNBUSDT": "BNB-USD",
    "ADAUSDT": "ADA-USD",
    "DOGEUSDT": "DOGE-USD",
    "XRPUSDT": "XRP-USD",
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "SOL": "SOL-USD"
}


def normalize_ticker(ticker: str) -> str:
    """Normaliza el ticker al formato adecuado de yfinance."""
    clean = ticker.upper().strip()
    return TICKER_MAP.get(clean, clean)


def generate_synthetic_market_data(ticker: str, periods: int = 500, interval_mins: int = 60) -> pd.DataFrame:
    """
    Genera datos sintéticos de mercado con Movimiento Browniano Geométrico y volatilidad estocástica.
    Garantiza que la defensa en vivo NUNCA falle si la conexión o API externa sufre timeout.
    """
    np.random.seed(42 + hash(ticker) % 1000)

    base_prices = {
        "BTCUSDT": 91500.0,
        "BTC-USD": 91500.0,
        "ETHUSDT": 3150.0,
        "ETH-USD": 3150.0,
        "SOLUSDT": 205.0,
        "SOL-USD": 205.0,
        "AAPL": 225.0,
        "NVDA": 142.0,
        "TSLA": 320.0,
        "SPY": 595.0
    }
    s0 = base_prices.get(ticker.upper(), 100.0)

    dt = 1.0 / (365.0 * 24.0)
    mu = 0.08  # Drift anualizado
    sigma = 0.45  # Volatilidad anualizada

    # Retornos aleatorios
    innovations = np.random.normal(0, 1, periods)
    returns = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * innovations

    # Saltos esporádicos (Jump Diffusion de Merton)
    jump_prob = 0.03
    jumps = np.random.choice([0, 1], size=periods, p=[1 - jump_prob, jump_prob])
    jump_sizes = np.random.normal(0, 0.02, periods) * jumps
    returns += jump_sizes

    close_prices = s0 * np.exp(np.cumsum(returns))

    # Creación de velas Open, High, Low, Close, Volume
    high_mult = 1.0 + np.abs(np.random.normal(0.003, 0.004, periods))
    low_mult = 1.0 - np.abs(np.random.normal(0.003, 0.004, periods))

    high_prices = close_prices * high_mult
    low_prices = close_prices * low_mult
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = s0

    # Asegurar consistencia de velas
    high_prices = np.maximum(high_prices, np.maximum(open_prices, close_prices))
    low_prices = np.minimum(low_prices, np.minimum(open_prices, close_prices))

    volume = np.random.lognormal(mean=12.0, sigma=0.8, size=periods)

    timestamps = pd.date_range(end=pd.Timestamp.now(), periods=periods, freq=f"{interval_mins}min")

    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": open_prices,
        "high": high_prices,
        "low": low_prices,
        "close": close_prices,
        "volume": volume
    })
    df.set_index("timestamp", inplace=True)
    return df


def fetch_historical_data(
    ticker: str = "BTCUSDT",
    period: str = "60d",
    interval: str = "1h",
    use_cache: bool = True
) -> pd.DataFrame:
    """
    Descarga datos de mercado reales de yfinance con manejo de errores robusto.
    Si yfinance no está disponible o falla por red/cuota, recurre al simulador Browniano.
    """
    clean_ticker = normalize_ticker(ticker)

    try:
        import yfinance as yf
        data = yf.download(clean_ticker, period=period, interval=interval, progress=False, auto_adjust=True)

        if data is not None and not data.empty and len(data) >= 30:
            # Si las columnas son MultiIndex (versión reciente de yfinance), aplanarlas
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [col[0].lower() for col in data.columns]
            else:
                data.columns = [col.lower() for col in data.columns]

            # Renombrar columnas si es necesario
            data = data.rename(columns={
                "adj close": "close"
            })
            required = ['open', 'high', 'low', 'close', 'volume']
            if all(col in data.columns for col in required):
                return data[required].dropna()
    except Exception as e:
        print(f"[DataFeed] Advertencia al obtener {ticker} vía yfinance: {e}. Usando generador estocástico de contingencia.")

    # Contingencia garantizada
    return generate_synthetic_market_data(ticker, periods=300, interval_mins=60)


def get_current_price(ticker: str) -> float:
    """Obtiene el precio más reciente del activo."""
    df = fetch_historical_data(ticker, period="2d", interval="1h")
    return float(df['close'].iloc[-1])

