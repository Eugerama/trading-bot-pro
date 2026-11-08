"""
Módulo de Gestión de Watchlist Dinámica y Filtro de Horario Bursátil.
Permite agregar y quitar tickers en caliente para que el escáner del bot
los monitoree y ejecute trades automáticamente.
Persiste la lista en 'watchlist.json'.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import pytz

logger = logging.getLogger("WatchlistManager")

WATCHLIST_FILE = "watchlist.json"

DEFAULT_WATCHLIST = [
    {"ticker": "BTCUSDT", "type": "CRYPTO", "added_at": "2026-11-01 00:00:00"},
    {"ticker": "ETHUSDT", "type": "CRYPTO", "added_at": "2026-11-01 00:00:00"},
    {"ticker": "SOLUSDT", "type": "CRYPTO", "added_at": "2026-11-01 00:00:00"},
    {"ticker": "NVDA", "type": "STOCK", "added_at": "2026-11-01 00:00:00"},
    {"ticker": "AAPL", "type": "STOCK", "added_at": "2026-11-01 00:00:00"}
]

STOCK_LIST = [
    "AAPL", "NVDA", "TSLA", "MSFT", "AMZN", "GOOGL", "GOOG", "META",
    "AMD", "INTC", "SPY", "QQQ", "DIA", "JPM", "BAC", "V", "DIS", "WMT"
]


class WatchlistManager:
    """Administra la lista de seguimiento activa para el motor de trading."""

    def __init__(self, filepath: str = WATCHLIST_FILE):
        self.filepath = filepath
        self.items: List[Dict[str, Any]] = []
        self.load()

    def load(self):
        """Carga la lista desde el disco o inicializa con los valores por defecto."""
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r") as f:
                    data = json.load(f)
                    if isinstance(data, list) and len(data) > 0:
                        self.items = data
                        return
            except Exception as e:
                logger.warning(f"Error al leer {self.filepath}: {e}")

        self.items = DEFAULT_WATCHLIST.copy()
        self.save()

    def save(self):
        """Guarda la lista actual en disco."""
        try:
            with open(self.filepath, "w") as f:
                json.dump(self.items, f, indent=2)
        except Exception as e:
            logger.error(f"Error al guardar {self.filepath}: {e}")

    def get_tickers(self) -> List[str]:
        """Retorna solo los símbolos activos como strings."""
        return [item["ticker"] for item in self.items]

    def get_all(self) -> List[Dict[str, Any]]:
        """Retorna la lista completa con metadatos."""
        return self.items

    def determine_type(self, ticker: str) -> str:
        """Determina si un activo es criptomoneda o acción tradicional."""
        clean = ticker.upper().strip()
        if clean in STOCK_LIST or clean.startswith("NASDAQ:") or clean.startswith("NYSE:"):
            return "STOCK"
        if "USDT" in clean or clean in ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA"]:
            return "CRYPTO"
        return "STOCK" if len(clean) <= 4 else "CRYPTO"

    def normalize(self, raw_ticker: str) -> str:
        """Normaliza el ticker para evitar duplicados o formatos ambiguos."""
        t = raw_ticker.upper().strip()
        crypto_roots = ["BTC", "ETH", "SOL", "BNB", "ADA", "XRP", "DOGE", "AVAX", "DOT", "LINK", "LTC", "NEAR", "SUI", "PEPE", "SHIB"]
        if t in crypto_roots:
            return f"{t}USDT"
        return t

    def add_ticker(self, raw_ticker: str) -> Dict[str, Any]:
        """Añade un nuevo ticker a la Watchlist activa."""
        clean = self.normalize(raw_ticker)
        if not clean or len(clean) < 2:
            return {"success": False, "message": "Ticker inválido."}

        # Verificar si ya existe
        for item in self.items:
            if item["ticker"] == clean:
                return {"success": False, "message": f"{clean} ya está en la Watchlist."}

        asset_type = self.determine_type(clean)
        new_entry = {
            "ticker": clean,
            "type": asset_type,
            "added_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self.items.append(new_entry)
        self.save()
        logger.info(f"[Watchlist] Ticker añadido: {clean} ({asset_type})")
        return {"success": True, "ticker": clean, "type": asset_type, "message": f"{clean} añadido a la Watchlist."}

    def remove_ticker(self, raw_ticker: str) -> Dict[str, Any]:
        """Elimina un ticker de la Watchlist activa."""
        clean = self.normalize(raw_ticker)
        initial_len = len(self.items)
        self.items = [item for item in self.items if item["ticker"] != clean]

        if len(self.items) < initial_len:
            self.save()
            logger.info(f"[Watchlist] Ticker eliminado: {clean}")
            return {"success": True, "ticker": clean, "message": f"{clean} eliminado de la Watchlist."}
        return {"success": False, "message": f"{clean} no se encontraba en la Watchlist."}

    def is_market_open(self, ticker: str) -> Dict[str, Any]:
        """
        Verifica si el mercado de este activo está abierto actualmente.
        - Cripto: 24/7 abierto siempre.
        - Acciones Wall Street: Lunes a Viernes 9:30 AM a 4:00 PM EST (Nueva York).
        """
        asset_type = self.determine_type(ticker)
        if asset_type == "CRYPTO":
            return {"is_open": True, "market": "24/7 Crypto Exchange", "status": "ABIERTO (24/7)"}

        # Verificación de horario de Wall Street (Eastern Time)
        try:
            ny_tz = pytz.timezone("America/New_York")
            now_ny = datetime.now(ny_tz)
        except Exception:
            # Fallback sin pytz usando hora local aproximada
            now_ny = datetime.now()

        weekday = now_ny.weekday()  # 0: Lunes ... 4: Viernes, 5: Sábado, 6: Domingo
        is_weekend = weekday >= 5

        # Horario 9:30 a 16:00
        open_time = now_ny.replace(hour=9, minute=30, second=0, microsecond=0)
        close_time = now_ny.replace(hour=16, minute=0, second=0, microsecond=0)

        is_trading_hours = not is_weekend and (open_time <= now_ny <= close_time)

        if is_trading_hours:
            return {"is_open": True, "market": "Wall Street (NYSE/NASDAQ)", "status": "ABIERTO"}
        else:
            reason = "Fin de Semana" if is_weekend else ("Pre-mercado" if now_ny < open_time else "Post-mercado / Noche")
            return {
                "is_open": False,
                "market": "Wall Street (NYSE/NASDAQ)",
                "status": f"CERRADO ({reason})",
                "next_open": "Lunes 09:30 EST" if is_weekend else "Próxima sesión 09:30 EST"
            }

