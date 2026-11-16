"""
Configuración compartida de la gráfica (activo + temporalidad).

Telegram la modifica (/tf, /activo, /chart) y el Dashboard Web la consulta en
/api/chart-settings para actualizar la gráfica de TradingView en vivo.
Se persiste en JSON para que funcione aunque el bot y el servidor web corran
en procesos distintos (main.py vs web/server.py).
"""

import json
import os
import threading
import time
from typing import Dict, Any, Optional

# Temporalidades soportadas.
#  - tv: intervalo del widget de TradingView
#  - yf_interval / yf_period: descarga con yfinance (respeta los límites de historial de intradía)
#  - resample: regla de pandas si yfinance no ofrece la temporalidad nativa (4h)
TIMEFRAMES: Dict[str, Dict[str, Any]] = {
    "1m":  {"label": "1 minuto",   "tv": "1",   "yf_interval": "1m",  "yf_period": "5d",   "resample": None},
    "5m":  {"label": "5 minutos",  "tv": "5",   "yf_interval": "5m",  "yf_period": "30d",  "resample": None},
    "15m": {"label": "15 minutos", "tv": "15",  "yf_interval": "15m", "yf_period": "30d",  "resample": None},
    "30m": {"label": "30 minutos", "tv": "30",  "yf_interval": "30m", "yf_period": "30d",  "resample": None},
    "1h":  {"label": "1 hora",     "tv": "60",  "yf_interval": "1h",  "yf_period": "30d",  "resample": None},
    "4h":  {"label": "4 horas",    "tv": "240", "yf_interval": "1h",  "yf_period": "180d", "resample": "4h"},
    "1d":  {"label": "Diario",     "tv": "D",   "yf_interval": "1d",  "yf_period": "1y",   "resample": None},
    "1w":  {"label": "Semanal",    "tv": "W",   "yf_interval": "1wk", "yf_period": "5y",   "resample": None},
}

# Sinónimos que el usuario puede escribir en Telegram
_TF_ALIASES = {
    "1min": "1m", "5min": "5m", "15min": "15m", "30min": "30m",
    "60": "1h", "60m": "1h", "1hr": "1h", "h1": "1h", "1hora": "1h",
    "240": "4h", "4hr": "4h", "h4": "4h", "4horas": "4h",
    "d": "1d", "1day": "1d", "diario": "1d", "dia": "1d", "día": "1d", "daily": "1d",
    "w": "1w", "1wk": "1w", "semanal": "1w", "semana": "1w", "weekly": "1w",
}

DEFAULT_TICKER = os.getenv("DEFAULT_TICKER", "BTCUSDT")
DEFAULT_TIMEFRAME = "1h"


def normalize_timeframe(tf: Optional[str]) -> Optional[str]:
    """Convierte la entrada del usuario ('4H', '240', 'diario'...) a una clave de TIMEFRAMES, o None si no es válida."""
    if not tf:
        return None
    key = tf.strip().lower()
    key = _TF_ALIASES.get(key, key)
    return key if key in TIMEFRAMES else None


def normalize_chart_ticker(ticker: Optional[str]) -> Optional[str]:
    """Valida un ticker simple (letras, números, '-', '.', ':', '=', '^'), máx. 20 caracteres."""
    if not ticker:
        return None
    t = ticker.strip().upper()
    if not t or len(t) > 20:
        return None
    if not all(c.isalnum() or c in "-.:=^" for c in t):
        return None
    return t


class ChartSettings:
    """Estado compartido de la gráfica, persistido en JSON y seguro entre hilos."""

    def __init__(self, path: str = "chart_settings.json"):
        self.path = path
        self._lock = threading.Lock()

    def _defaults(self) -> Dict[str, Any]:
        return {
            "ticker": DEFAULT_TICKER,
            "timeframe": DEFAULT_TIMEFRAME,
            "updated_at": 0.0,
            "updated_by": "default",
        }

    def get(self) -> Dict[str, Any]:
        data = self._defaults()
        try:
            if os.path.exists(self.path):
                with open(self.path, "r", encoding="utf-8") as f:
                    stored = json.load(f)
                if isinstance(stored, dict):
                    data.update(stored)
        except Exception:
            pass
        # Saneamiento por si el JSON se editó a mano
        data["timeframe"] = normalize_timeframe(data.get("timeframe")) or DEFAULT_TIMEFRAME
        data["ticker"] = normalize_chart_ticker(data.get("ticker")) or DEFAULT_TICKER
        tf = TIMEFRAMES[data["timeframe"]]
        data["tv_interval"] = tf["tv"]
        data["timeframe_label"] = tf["label"]
        return data

    def update(self, ticker: Optional[str] = None, timeframe: Optional[str] = None, source: str = "telegram") -> Dict[str, Any]:
        """Actualiza activo y/o temporalidad. Lanza ValueError si algún valor no es válido."""
        new_tf = None
        new_tk = None
        if timeframe is not None:
            new_tf = normalize_timeframe(timeframe)
            if not new_tf:
                raise ValueError(f"Temporalidad no válida: '{timeframe}'. Opciones: {', '.join(TIMEFRAMES)}")
        if ticker is not None:
            new_tk = normalize_chart_ticker(ticker)
            if not new_tk:
                raise ValueError(f"Ticker no válido: '{ticker}'.")

        with self._lock:
            current = self.get()
            stored = {
                "ticker": new_tk or current["ticker"],
                "timeframe": new_tf or current["timeframe"],
                "updated_at": time.time(),
                "updated_by": source,
            }
            tmp_path = f"{self.path}.tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(stored, f)
            os.replace(tmp_path, self.path)  # escritura atómica
        return self.get()


def fetch_chart_data(ticker: str, timeframe: str):
    """Descarga velas para la temporalidad indicada (re-muestreando a 4h si hace falta)."""
    from core.data_feed import fetch_historical_data

    tf_key = normalize_timeframe(timeframe) or DEFAULT_TIMEFRAME
    cfg = TIMEFRAMES[tf_key]
    df = fetch_historical_data(ticker, period=cfg["yf_period"], interval=cfg["yf_interval"])
    if cfg["resample"] and df is not None and len(df) > 0:
        df = df.resample(cfg["resample"]).agg({
            "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
        }).dropna()
    return df


chart_settings = ChartSettings()
