"""
Punto de Entrada Principal del Proyecto Integrador.
Ejecuta concurrentemente:
1. Servidor Web FastAPI (Dashboard interactivo con TradingView y autenticación)
2. Motor de Trading Cuantitativo 24/7 en segundo plano
3. Bot de ChatOps de Telegram con monitoreo de la VM de Compute Engine
"""

import os
import sys
import time
import logging
import threading
from datetime import datetime
from dotenv import load_dotenv
import uvicorn

# Cargar variables de entorno locales si existen
load_dotenv()

# Configurar logging (ocultando logs de httpx para no exponer el token de Telegram)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("trading_bot.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logger = logging.getLogger("MainOrchestrator")

from core.data_feed import fetch_historical_data, get_current_price
from core.indicators import compute_all_indicators
from core.ml_engine import QuantitativeMLEngine
from core.strategy import HybridQuantStrategy
from core.risk_manager import RiskManager
from core.execution_engine import ExecutionEngine
from core.gemini_agent import GeminiQuantAgent
from core.watchlist_manager import WatchlistManager
from telegram_bot.bot import TelegramTradingBot
from web.server import app, execution_engine, ml_engine, telegram_bot, watchlist_mgr


def background_trading_loop(interval_seconds: int = 45):
    """
    Loop continuo y autónomo de trading cuantitativo en la VM de Google Cloud.
    Escaner 24/7 sobre la Watchlist dinámica:
    1. Lee los tickers activos configurados por el usuario en la Watchlist.
    2. Respeta el horario de mercado (Wall Street vs Cripto 24/7).
    3. Monitorea y actualiza posiciones abiertas (Trailing / Stop Loss / Take Profit).
    4. Detecta confluencias con Random Forest y abre posiciones automáticamente.
    5. Envía alertas push en tiempo real a Telegram y guarda en trading_bot.log.
    """
    logger.info("Iniciando escáner y motor de trading autónomo 24/7 sobre Watchlist dinámica...")
    strat = HybridQuantStrategy(rf_confidence_threshold=0.45)
    risk = RiskManager()

    while True:
        try:
            auto_trade_enabled = os.getenv("AUTO_TRADING_ENABLED", "true").lower() == "true"
            # Cargar los tickers dinámicos de la Watchlist del usuario
            tracked_tickers = watchlist_mgr.get_tickers()

            for ticker in tracked_tickers:
                # 1. Obtener precio actual
                current_price = get_current_price(ticker)

                # 2. Verificar y actualizar posiciones activas existentes (Trailing SL / TP)
                closed_trades = execution_engine.update_and_check_positions(ticker, current_price)
                for closed in closed_trades:
                    msg = (
                        f"🔔 *ALERTA DE CIERRE DE POSICIÓN*\n"
                        f"• Ticker: *{closed['ticker']}*\n"
                        f"• Motivo: `{closed['exit_reason']}`\n"
                        f"• Salida: `${closed['exit_price']:,.2f}`\n"
                        f"• PnL Realizado: `${closed['pnl']:+,.2f} ({closed['roi_pct']:+.2f}%)`"
                    )
                    logger.info(f"Posición cerrada en {ticker}: {closed['exit_reason']} PnL: ${closed['pnl']}")
                    if telegram_bot.app and telegram_bot.is_running:
                        import asyncio
                        asyncio.run_coroutine_threadsafe(
                            telegram_bot.send_alert(msg),
                            telegram_bot._loop
                        )

                # 2.5 Cierre inteligente por reversión de tendencia (asegura ganancias si el mercado gira)
                if auto_trade_enabled and ticker in execution_engine.active_positions:
                    pos = execution_engine.active_positions[ticker]
                    if pos.get("roi_pct", 0.0) >= 0.35:
                        try:
                            df_rev = fetch_historical_data(ticker, period="30d", interval="1h")
                            ml_rev = ml_engine.predict_latest(df_rev, ticker=ticker)
                            eval_rev = strat.evaluate_signals(df_rev, ml_rev)
                            rev_sig = eval_rev["signal"]
                            if (pos["side"] == "BUY" and rev_sig == "SELL") or (pos["side"] == "SELL" and rev_sig == "BUY"):
                                closed_rev = execution_engine.close_position(ticker, current_price, "REVERSION_TENDENCIA_PROFIT")
                                if closed_rev:
                                    rev_msg = (
                                        f"🎯 *TOMA DE GANANCIA POR REVERSIÓN DE TENDENCIA*\n"
                                        f"• Ticker: *{ticker}* | Lado previo: `{closed_rev['side']}`\n"
                                        f"• Salida: `${closed_rev['exit_price']:,.2f}`\n"
                                        f"• PnL Asegurado: `${closed_rev['pnl']:+,.2f} ({closed_rev['roi_pct']:+.2f}%)`"
                                    )
                                    logger.info(f"Posición cerrada por reversión en {ticker}: PnL ${closed_rev['pnl']}")
                                    if telegram_bot.app and telegram_bot.is_running:
                                        import asyncio
                                        asyncio.run_coroutine_threadsafe(
                                            telegram_bot.send_alert(rev_msg),
                                            telegram_bot._loop
                                        )
                        except Exception:
                            pass

                # 3. ESCÁNER AUTÓNOMO DE NUEVAS OPORTUNIDADES
                # Si el auto-trading está activo y NO hay posición abierta para este ticker:
                if auto_trade_enabled and ticker not in execution_engine.active_positions:
                    # Filtro de horario bursátil institucional:
                    # Si es una acción (NYSE/NASDAQ) y el mercado está cerrado, omitir apertura automática
                    market_status = watchlist_mgr.is_market_open(ticker)
                    if not market_status["is_open"]:
                        logger.debug(f"[Escáner] {ticker} en pausa: Mercado de Wall Street cerrado ({market_status['status']}).")
                        continue

                    try:
                        df = fetch_historical_data(ticker, period="30d", interval="1h")
                        ml_pred = ml_engine.predict_latest(df, ticker=ticker)
                        strat_eval = strat.evaluate_signals(df, ml_pred)
                        signal = strat_eval["signal"]

                        # Si hay señal decisiva (BUY o SELL) con alta probabilidad en el modelo
                        if signal in ["BUY", "SELL"]:
                            trade_risk = risk.calculate_trade_parameters(
                                balance=execution_engine.balance,
                                current_price=current_price,
                                atr=ml_pred["atr_14"],
                                signal=signal
                            )

                            if trade_risk.get("valid"):
                                res = execution_engine.open_position(
                                    ticker=ticker,
                                    side=signal,
                                    entry_price=current_price,
                                    size=trade_risk["position_size_units"],
                                    stop_loss=trade_risk["stop_loss"],
                                    take_profit=trade_risk["take_profit"]
                                )

                                if res.get("success"):
                                    logger.info(f"🤖 [AUTO-TRADE] Orden {signal} ejecutada para {ticker} @ ${current_price:,.2f}")
                                    alert_msg = (
                                        f"🤖 *AUTO-TRADE EJECUTADO EN VM (24/7)*\n"
                                        f"━━━━━━━━━━━━━━━━━━━━━━\n"
                                        f"• Activo (Watchlist): *{ticker}*\n"
                                        f"• Señal: `{'🟢 BUY' if signal == 'BUY' else '🔴 SELL'}` (Random Forest: {ml_pred['confidence']*100:.1f}%)\n"
                                        f"• Entrada: `${current_price:,.2f}`\n"
                                        f"• Tamaño: `{trade_risk['position_size_units']}` unidades (${trade_risk['position_value_usd']:,.2f})\n"
                                        f"• Stop Loss (ATR): `${trade_risk['stop_loss']:,.2f}` (-{trade_risk['sl_distance_pct']}%)\n"
                                        f"• Take Profit (ATR): `${trade_risk['take_profit']:,.2f}` (+{trade_risk['tp_distance_pct']}%)\n"
                                        f"• R/B Ratio: `{trade_risk['risk_reward_ratio']}`\n"
                                        f"• Motivo: _{strat_eval['reason']}_\n"
                                        f"━━━━━━━━━━━━━━━━━━━━━━\n"
                                        f"💡 _Notificación automática enviada desde GCP Compute Engine._"
                                    )
                                    if telegram_bot.app and telegram_bot.is_running:
                                        import asyncio
                                        asyncio.run_coroutine_threadsafe(
                                            telegram_bot.send_alert(alert_msg),
                                            telegram_bot._loop
                                        )
                    except Exception as scan_err:
                        logger.debug(f"Error escaneando {ticker}: {scan_err}")

            time.sleep(interval_seconds)
        except Exception as e:
            logger.error(f"Error en loop de trading autónomo: {e}")
            time.sleep(10)


def main():
    host = os.getenv("WEB_HOST", "0.0.0.0")
    port = int(os.getenv("WEB_PORT", "8080"))

    print("=" * 70)
    print("🚀 QUANTITATIVE TRADING BOT & AI DASHBOARD (GCP COMPUTE ENGINE)")
    print(f"📡 Servidor Web: http://{host}:{port}")
    print(f"🔒 Credenciales por defecto: admin / cuantitativo2026")
    print(f"🤖 Bot de Telegram: {'Configurado' if os.getenv('TELEGRAM_BOT_TOKEN') else 'Esperando token en .env'}")
    print("=" * 70)

    # Iniciar trading loop en hilo daemon
    trading_thread = threading.Thread(target=background_trading_loop, args=(45,), daemon=True)
    trading_thread.start()

    # Iniciar servidor FastAPI con Uvicorn
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()

