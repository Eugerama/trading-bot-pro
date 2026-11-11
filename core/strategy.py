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

        # 2. MACD
        if metrics_cfg.get("use_macd", True):
            if macd > macd_signal:
                tech_score += 1
                tech_reasons.append("Cruce alcista MACD")
            else:
                tech_score -= 1
                tech_reasons.append("Cruce bajista MACD")

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
                tech_score += 1
            elif stoch_k < stoch_d and stoch_k > 20:
                tech_score -= 1

        # 5. Señal de Machine Learning (si está habilitada)
        if metrics_cfg.get("use_ml", True):
            ml_signal = ml_prediction.get("signal", "HOLD")
            ml_conf = ml_prediction.get("confidence", 0.0)
        else:
            ml_signal = "HOLD"
            ml_conf = 0.50

        # Confluencia base
        final_signal = "HOLD"
        reason = "Consolidación de indicadores sin consenso decisivo."

        threshold = self.rf_confidence_threshold
        if metrics_cfg.get("use_ml", True):
            if ml_signal == "BUY" and tech_score >= 1 and ml_conf >= threshold:
                final_signal = "BUY"
                reason = f"Confluencia Alcista: RF predice BUY ({ml_conf*100:.1f}%) con confirmación técnica ({', '.join(tech_reasons)})."
            elif ml_signal == "SELL" and tech_score <= -1 and ml_conf >= threshold:
                final_signal = "SELL"
                reason = f"Confluencia Bajista: RF predice SELL ({ml_conf*100:.1f}%) con debilidad técnica ({', '.join(tech_reasons)})."
            elif tech_score >= 3 and ml_signal != "SELL":
                final_signal = "BUY"
                reason = f"Impulso Técnico Fuerte (+{tech_score}): Alineación total ({', '.join(tech_reasons)}) con RF neutral."
            elif tech_score <= -3 and ml_signal != "BUY":
                final_signal = "SELL"
                reason = f"Impulso Técnico Bajista ({tech_score}): Alineación bajista ({', '.join(tech_reasons)}) con RF neutral."
        else:
            # Si el usuario desactivó ML, decide puramente por indicadores técnicos
            if tech_score >= 2:
                final_signal = "BUY"
                reason = f"Confluencia Técnica Alcista: {', '.join(tech_reasons)}."
            elif tech_score <= -2:
                final_signal = "SELL"
                reason = f"Confluencia Técnica Bajista: {', '.join(tech_reasons)}."

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
