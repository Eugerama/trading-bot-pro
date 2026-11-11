"""
Módulo de Estrategia Cuantitativa Híbrida.
Combina:
- Indicadores técnicos seleccionables por el usuario (RSI, MACD, Regresión, SAR, Estocástico)
- Modelo predictivo Random Forest
- Filtro de Procesamiento de Lenguaje Natural (NLP) con Gemini Enterprise
"""

from typing import Dict, Any, Optional
import pandas as pd

from core.indicators import compute_all_indicators
from core.monte_carlo import run_monte_carlo_simulation


class HybridQuantStrategy:
    """Estrategia algorítmica que fusiona análisis técnico clásico, Machine Learning y NLP."""

    def __init__(
        self,
        rsi_oversold: float = 35.0,
        rsi_overbought: float = 65.0,
        rf_confidence_threshold: float = 0.45
    ):
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.rf_confidence_threshold = rf_confidence_threshold

    def evaluate_signals(
        self,
        df_raw: pd.DataFrame,
        ml_prediction: Dict[str, Any],
        nlp_data: Optional[Dict[str, Any]] = None,
        enabled_metrics: Optional[Dict[str, bool]] = None
    ) -> Dict[str, Any]:
        """
        Evalúa el estado del mercado considerando las métricas activadas por el usuario
        y las señales combinadas de Machine Learning y NLP Sentiment.
        """
        # Métricas activadas por defecto si el usuario no especificó preferencias
        metrics_cfg = {
            "use_rsi": True,
            "use_macd": True,
            "use_sar": True,
            "use_lr": True,
            "use_stoch": True,
            "use_montecarlo": True,
            "use_ml": True,
            "use_nlp": True
        }
        if enabled_metrics:
            metrics_cfg.update(enabled_metrics)

        df = compute_all_indicators(df_raw)
        last_row = df.iloc[-1]

        close = float(last_row['close'])
        rsi = float(last_row['rsi_14'])
        macd = float(last_row['macd'])
        macd_signal = float(last_row['macd_signal'])
        sar = float(last_row['parabolic_sar'])
        lr_slope = float(last_row['lr_slope'])
        lr_r2 = float(last_row['lr_r2'])
        ema_21 = float(last_row['ema_21'])
        ema_50 = float(last_row['ema_50'])
        stoch_k = float(last_row['stoch_k'])
        stoch_d = float(last_row['stoch_d'])
        atr = float(last_row['atr_14'])

        # Evaluación técnica ponderada según métricas activas
        tech_score = 0
        tech_reasons = []

        # 1. Regresión Lineal y SAR
        if metrics_cfg.get("use_sar", True) and metrics_cfg.get("use_lr", True):
            if close > sar and lr_slope > 0:
                tech_score += 2
                tech_reasons.append("SAR alcista y pendiente positiva")
            elif close < sar and lr_slope < 0:
                tech_score -= 2
                tech_reasons.append("SAR bajista y pendiente negativa")
        elif metrics_cfg.get("use_sar", True):
            if close > sar:
                tech_score += 1
            else:
                tech_score -= 1
        elif metrics_cfg.get("use_lr", True):
            if lr_slope > 0:
                tech_score += 1
            else:
                tech_score -= 1

        # 2. MACD con decaimiento exponencial/escalonado según antigüedad del cruce
        if metrics_cfg.get("use_macd", True):
            macd_diff = df['macd'] - df['macd_signal']
            current_bullish = macd > macd_signal
            bars_since_cross = 1
            for i in range(len(macd_diff) - 2, -1, -1):
                was_bullish = float(macd_diff.iloc[i]) > 0
                if was_bullish == current_bullish:
                    bars_since_cross += 1
                else:
                    break

            if bars_since_cross <= 2:
                macd_weight = 1.00
                cross_label = f"recién cruzado hace {bars_since_cross}v (+1.00)"
            elif bars_since_cross <= 5:
                macd_weight = 0.70
                cross_label = f"cruce reciente hace {bars_since_cross}v ({0.70:.2f})"
            elif bars_since_cross <= 10:
                macd_weight = 0.40
                cross_label = f"hace {bars_since_cross}v ({0.40:.2f})"
            else:
                macd_weight = 0.15
                cross_label = f"maduro hace {bars_since_cross}v ({0.15:.2f})"

            if current_bullish:
                tech_score = round(tech_score + macd_weight, 2)
                tech_reasons.append(f"MACD alcista ({cross_label})")
            else:
                tech_score = round(tech_score - macd_weight, 2)
                tech_reasons.append(f"MACD bajista ({cross_label})")

        # 3. RSI
        if metrics_cfg.get("use_rsi", True):
            if rsi < self.rsi_oversold:
                tech_score += 1
                tech_reasons.append(f"RSI en sobreventa ({rsi:.1f})")
            elif rsi > self.rsi_overbought:
                tech_score -= 1
                tech_reasons.append(f"RSI en sobrecompra ({rsi:.1f})")

        # 4. Estocástico
        if metrics_cfg.get("use_stoch", True):
            if stoch_k > stoch_d and stoch_k < 80:
                tech_score = round(tech_score + 1.0, 2)
            elif stoch_k < stoch_d and stoch_k > 20:
                tech_score = round(tech_score - 1.0, 2)

        # 4.5. Simulación Monte Carlo integrada al Score Técnico (+0.50 / -0.50)
        if metrics_cfg.get("use_montecarlo", True):
            try:
                eff_atr = max(atr, close * 0.005)
                tentative_side = "BUY" if tech_score >= 0 else "SELL"
                if tentative_side == "BUY":
                    mc_sl = close - (eff_atr * 1.5)
                    mc_tp = close + (eff_atr * 2.5)
                else:
                    mc_sl = close + (eff_atr * 1.5)
                    mc_tp = close - (eff_atr * 2.5)

                mc_eval = run_monte_carlo_simulation(
                    df_raw, close, mc_sl, mc_tp,
                    position_type=tentative_side,
                    num_simulations=500,
                    time_horizon_bars=36
                )
                exp_ret = mc_eval.get("expected_return", 0.0)
                prob_tp = mc_eval.get("prob_take_profit", 0.0) * 100.0

                if exp_ret > 0:
                    if tentative_side == "BUY":
                        tech_score = round(tech_score + 0.50, 2)
                        tech_reasons.append(f"Monte Carlo favorable P(TP)={prob_tp:.0f}% (+0.50)")
                    else:
                        tech_score = round(tech_score - 0.50, 2)
                        tech_reasons.append(f"Monte Carlo bajista P(TP)={prob_tp:.0f}% (-0.50)")
                elif exp_ret < -0.25:
                    # Si la deriva estocástica contradice la dirección tentativa, resta fuerza
                    if tentative_side == "BUY":
                        tech_score = round(tech_score - 0.25, 2)
                    else:
                        tech_score = round(tech_score + 0.25, 2)
            except Exception:
                pass

        # 5. Señal de Machine Learning (si está habilitada)
        if metrics_cfg.get("use_ml", True):
            ml_signal = ml_prediction.get("signal", "HOLD")
            ml_conf = ml_prediction.get("confidence", 0.0)
        else:
            ml_signal = "HOLD"
            ml_conf = 0.50

        # Confluencia base con Filtro de Tendencia (EMA 21) y Filtro Anti-Lateral (R² >= 0.15)
        final_signal = "HOLD"
        reason = "Consolidación de indicadores sin consenso decisivo."

        threshold = self.rf_confidence_threshold
        trend_up = close >= ema_21
        trend_down = close <= ema_21
        has_trend_quality = lr_r2 >= 0.15

        if metrics_cfg.get("use_ml", True):
            if has_trend_quality and trend_up and ml_signal == "BUY" and tech_score >= 1.3 and ml_conf >= threshold:
                final_signal = "BUY"
                reason = f"Confluencia Alcista (R²={lr_r2:.2f}, Precio>EMA21): RF BUY ({ml_conf*100:.1f}%) + Técnico ({', '.join(tech_reasons)})."
            elif has_trend_quality and trend_down and ml_signal == "SELL" and tech_score <= -1.3 and ml_conf >= threshold:
                final_signal = "SELL"
                reason = f"Confluencia Bajista Simétrica (R²={lr_r2:.2f}, Precio<EMA21): RF SELL ({ml_conf*100:.1f}%) + Técnico ({', '.join(tech_reasons)})."
            elif has_trend_quality and trend_up and tech_score >= 2.65 and ml_signal != "SELL":
                final_signal = "BUY"
                reason = f"Impulso Técnico Fuerte (+{tech_score:.2f}, R²={lr_r2:.2f}): Alineación alcista ({', '.join(tech_reasons)})."
            elif has_trend_quality and trend_down and tech_score <= -2.65 and ml_signal != "BUY":
                final_signal = "SELL"
                reason = f"Impulso Técnico Bajista ({tech_score:.2f}, R²={lr_r2:.2f}): Alineación bajista ({', '.join(tech_reasons)})."
        else:
            # Si el usuario desactivó ML, decide puramente por indicadores técnicos y filtro de tendencia
            if has_trend_quality and trend_up and tech_score >= 2.0:
                final_signal = "BUY"
                reason = f"Confluencia Técnica Alcista (R²={lr_r2:.2f}): {', '.join(tech_reasons)}."
            elif has_trend_quality and trend_down and tech_score <= -2.0:
                final_signal = "SELL"
                reason = f"Confluencia Técnica Bajista (R²={lr_r2:.2f}): {', '.join(tech_reasons)}."

        # 6. FILTRO DE SENTIMIENTO NLP (GEMINI ENTERPRISE)
        nlp_score = 0.0
        nlp_class = "NEUTRAL"
        if metrics_cfg.get("use_nlp", True) and nlp_data:
            nlp_score = nlp_data.get("sentiment_score", 0.0)
            nlp_class = nlp_data.get("classification", "NEUTRAL")

            # Regla de Veto Institucional por Sentimiento Adverso
            if final_signal == "BUY" and nlp_data.get("trade_veto_buy"):
                final_signal = "HOLD"
                reason = f"🛡️ Veto NLP Gemini: Aunque los indicadores técnicos eran alcistas, el sentimiento de noticias es muy bajista ({nlp_score:+.2f}). Trade bloqueado por gestión de riesgo."
            elif final_signal == "SELL" and nlp_data.get("trade_veto_sell"):
                final_signal = "HOLD"
                reason = f"🛡️ Veto NLP Gemini: Sentimiento de noticias eufórico ({nlp_score:+.2f}) bloquea posición corta (SELL)."
            elif final_signal in ["BUY", "SELL"] and nlp_class in ["ALCISTA", "MUY ALCISTA", "BAJISTA", "MUY BAJISTA"]:
                reason += f" | NLP Gemini respalda con sentimiento {nlp_class} ({nlp_score:+.2f})."

        return {
            "signal": final_signal,
            "reason": reason,
            "tech_score": tech_score,
            "ml_signal": ml_signal,
            "ml_confidence": ml_conf,
            "nlp_sentiment_score": nlp_score,
            "nlp_classification": nlp_class,
            "active_metrics": metrics_cfg,
            "metrics": {
                "close": close,
                "rsi_14": round(rsi, 2),
                "macd": round(macd, 4),
                "macd_signal": round(macd_signal, 4),
                "stoch_k": round(stoch_k, 2),
                "stoch_d": round(stoch_d, 2),
                "lr_slope": round(lr_slope, 6),
                "sar": round(sar, 4),
                "atr_14": round(atr, 4)
            }
        }
