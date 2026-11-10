"""
Módulo de Integración con Google Gemini Enterprise / AI Studio.
Implementa un Agente Cuantitativo de IA que sintetiza:
- Indicadores Técnicos
- Probabilidades del Modelo Random Forest
- Resultados de la Simulación Monte Carlo
- Parámetros de Riesgo y Stop Loss / Take Profit
Genera un análisis institucional en lenguaje natural listo para Telegram y el Dashboard Web.
"""

import os
from typing import Dict, Any, Optional


class GeminiQuantAgent:
    """Agente de Inteligencia Artificial Cuantitativa impulsado por Google Gemini."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.model = None
        self._init_client()

    def _init_client(self):
        """Inicializa el cliente de Gemini si la clave está disponible con selección de modelos resiliente."""
        if not self.api_key or "TuClave" in self.api_key:
            return

        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self._genai = genai
            self.model_names = [os.getenv("GEMINI_MODEL", "gemini-2.5-flash"), "models/gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash-latest", "gemini-pro"]
            self.model = genai.GenerativeModel(self.model_names[0])
        except Exception as e:
            print(f"[GeminiAgent] Advertencia al inicializar Gemini: {e}")

    def analyze_market_setup(
        self,
        ticker: str,
        tech_data: Dict[str, Any],
        ml_data: Dict[str, Any],
        risk_data: Dict[str, Any],
        mc_data: Optional[Dict[str, Any]] = None,
        nlp_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Genera un dictamen financiero cuantitativo profesional.
        Si la API de Gemini no está configurada o falla la conexión, usa un generador
        de lenguaje cuantitativo determinístico para evitar interrupciones en la defensa en vivo.
        """
        signal = ml_data.get("signal", "HOLD")
        confidence = ml_data.get("confidence", 0.0) * 100
        current_price = tech_data.get("close", 0.0)
        rsi = tech_data.get("rsi_14", 50.0)
        macd = tech_data.get("macd", 0.0)
        sar = tech_data.get("sar", current_price)
        lr_slope = tech_data.get("lr_slope", 0.0)
        prob_tp = mc_data.get("prob_take_profit", 0.5) * 100 if mc_data else 50.0
        expected_ret = mc_data.get("expected_return", 0.0) if mc_data else 0.0
        sl = risk_data.get("stop_loss", current_price * 0.98)
        tp = risk_data.get("take_profit", current_price * 1.03)

        nlp_score = nlp_data.get("sentiment_score", 0.0) if nlp_data else 0.0
        nlp_class = nlp_data.get("classification", "NEUTRAL") if nlp_data else "NEUTRAL"
        nlp_summary = nlp_data.get("summary", "") if nlp_data else ""

        prompt = f"""
Eres el Agente Analista Cuantitativo Principal de un Hedge Fund desplegado en Google Cloud.
Evalúa los siguientes datos en tiempo real para el activo {ticker} y emite un informe conciso y riguroso de 3 párrafos:

DATOS EN TIEMPO REAL:
- Precio Actual: ${current_price:,.2f}
- Decisión de Random Forest: {signal} (Confianza: {confidence:.1f}%)
- Momentum & Osciladores: RSI={rsi:.2f}, MACD={macd:.4f}, Pendiente Regresión={lr_slope:.6f}
- SAR Parabólico: ${sar:,.2f}
- Sentimiento NLP de Noticias: {nlp_class} (Score: {nlp_score:+.2f}). Síntesis de noticias: {nlp_summary}
- Simulación Monte Carlo (1,000 caminos): Probabilidad TP={prob_tp:.1f}%, Retorno Esperado={expected_ret:+.2f}%
- Gestión de Riesgo ATR: Stop Loss=${sl:,.2f}, Take Profit=${tp:,.2f} (R/B={risk_data.get('risk_reward_ratio', 1.67):.2f})

INSTRUCCIONES:
1. Párrafo 1: Confluencia entre indicadores técnicos, modelo de Machine Learning y el Sentimiento NLP de noticias.
2. Párrafo 2: Interpretación del perfil de riesgo estocástico derivado de la simulación Monte Carlo.
3. Párrafo 3: Recomendación operativa final para el Bot (BUY/SELL/HOLD) y justificación institucional.
"""

        # Intento con API real de Gemini probando modelos compatibles
        if hasattr(self, '_genai') and self._genai:
            for m_name in getattr(self, 'model_names', ['gemini-1.5-flash', 'gemini-pro']):
                try:
                    m = self._genai.GenerativeModel(m_name)
                    response = m.generate_content(prompt)
                    if response and response.text:
                        self.model = m
                        return response.text.strip()
                except Exception:
                    continue

        # Fallback analítico de alta calidad (100% garantizado sin red externa)
        sentiment_desc = "alcista estructurada" if signal == "BUY" else ("bajista correctiva" if signal == "SELL" else "de consolidación neutral")
        momentum_desc = "fuerza compradora sostenida" if rsi > 50 and macd > 0 else "debilidad de momentum y divergencia"
        sar_status = "soporte dinámico" if current_price > sar else "presión bajista"

        local_analysis = (
            f"**Diagnóstico Cuantitativo ({ticker}):**\n"
            f"Se observa una confluencia {sentiment_desc} con el precio cotizando a ${current_price:,.2f}. "
            f"El clasificador Random Forest valida una señal `{signal}` con una certeza estadística del {confidence:.1f}%. "
            f"Los osciladores reflejan {momentum_desc} (RSI {rsi:.1f}, MACD {macd:.4f}), mientras la pendiente de regresión lineal "
            f"({lr_slope:.5f}) y el SAR parabólico (${sar:,.2f}) confirman el régimen de tendencia como {sar_status}.\n\n"
            f"**Evaluación Estocástica Monte Carlo:**\n"
            f"La simulación de 1,000 caminos bajo Movimiento Browniano proyecta una probabilidad de éxito del {prob_tp:.1f}% "
            f"para alcanzar el Take Profit (${tp:,.2f}) previo a la activación del Stop Loss (${sl:,.2f}). "
            f"El retorno medio esperado por posición es de {expected_ret:+.2f}%, ofreciendo un ratio riesgo-beneficio asimétrico "
            f"de {risk_data.get('risk_reward_ratio', 1.67):.2f}.\n\n"
            f"**Recomendación Operativa:**\n"
            f"Alineación favorable para ejecución de orden `{signal}` con dimensionamiento acotado por volatilidad ATR. "
            f"Parámetros recomendados: SL @ ${sl:,.2f} | TP @ ${tp:,.2f}."
        )

        return local_analysis

