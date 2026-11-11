"""
Módulo de Gestión Cuantitativa de Riesgo (Risk Management).
Implementa:
- Dimensionamiento de posición dinámico por volatilidad ATR y % de riesgo de cuenta
- Stop Loss y Take Profit dinámicos basados en múltiplos de ATR
- Circuit breaker de Max Drawdown para salvaguardar el capital en la VM
"""

from typing import Dict, Any


class RiskManager:
    """Gestiona el riesgo y dimensionamiento de órdenes en tiempo real."""

    def __init__(
        self,
        risk_per_trade_pct: float = 0.02,     # 2% de capital arriesgado por trade
        atr_sl_mult: float = 1.5,             # Multiplicador ATR para Stop Loss
        atr_tp_mult: float = 2.5,             # Multiplicador ATR para Take Profit
        max_drawdown_limit: float = 0.15      # 15% Max Drawdown Circuit Breaker
    ):
        self.risk_per_trade_pct = risk_per_trade_pct
        self.atr_sl_mult = atr_sl_mult
        self.atr_tp_mult = atr_tp_mult
        self.max_drawdown_limit = max_drawdown_limit

    def calculate_trade_parameters(
        self,
        balance: float,
        current_price: float,
        atr: float,
        signal: str,
        ticker: str = ""
    ) -> Dict[str, Any]:
        """
        Calcula el Stop Loss, Take Profit, tamaño de lote y monto monetario de riesgo.
        Adapta el espacio de ATR automáticamente si es Cripto (2.0x SL / 3.2x TP) vs Acciones (1.5x SL / 2.5x TP).
        """
        if signal not in ["BUY", "SELL"]:
            return {
                "valid": False,
                "reason": "Señal neutral (HOLD) - No se requiere orden de riesgo."
            }

        # Asegurar un ATR mínimo positivo
        effective_atr = max(atr, current_price * 0.005)

        clean_tk = ticker.upper().strip() if ticker else ""
        is_crypto = "USDT" in clean_tk or "-USD" in clean_tk or clean_tk in ["BTC", "ETH", "SOL", "BNB", "ADA", "XRP", "DOGE"]

        sl_mult = max(self.atr_sl_mult, 2.0) if is_crypto else self.atr_sl_mult
        tp_mult = max(self.atr_tp_mult, 3.2) if is_crypto else self.atr_tp_mult

        sl_distance = effective_atr * sl_mult
        tp_distance = effective_atr * tp_mult

        if signal == "BUY":
            stop_loss = current_price - sl_distance
            take_profit = current_price + tp_distance
        else:  # SELL
            stop_loss = current_price + sl_distance
            take_profit = current_price - tp_distance

        risk_capital = balance * self.risk_per_trade_pct
        price_distance = abs(current_price - stop_loss)

        if price_distance <= 1e-6:
            units = 0.01
        else:
            units = risk_capital / price_distance

        # Limitar el apalancamiento efectivo máximo a 3x del balance total
        max_units = (balance * 3.0) / current_price
        units = min(units, max_units)

        # Redondeo razonable para cripto/acciones
        position_size = round(units, 4)
        position_value_usd = round(position_size * current_price, 2)
        risk_reward_ratio = round(tp_distance / sl_distance, 2)

        return {
            "valid": True,
            "signal": signal,
            "entry_price": round(current_price, 4),
            "stop_loss": round(stop_loss, 4),
            "take_profit": round(take_profit, 4),
            "sl_distance_pct": round((sl_distance / current_price) * 100, 2),
            "tp_distance_pct": round((tp_distance / current_price) * 100, 2),
            "risk_reward_ratio": risk_reward_ratio,
            "position_size_units": position_size,
            "position_value_usd": position_value_usd,
            "max_risk_usd": round(risk_capital, 2)
        }

    def check_circuit_breaker(self, current_balance: float, peak_balance: float) -> bool:
        """Verifica si el drawdown acumulado ha excedido el límite de seguridad."""
        if peak_balance <= 0:
            return False
        drawdown = (peak_balance - current_balance) / peak_balance
        return drawdown >= self.max_drawdown_limit

