# GUÍA TÁCTICA PARA LA DEFENSA EN VIVO (19 DE NOVIEMBRE DE 2026)
## Protocolo Anti-Penalizaciones y Demostración Operativa en Vivo

---

### ⚠️ 1. CHECKLIST PREVIO OBLIGATORIO (EVITAR PENALIZACIÓN DE -10 Y -15 PUNTOS)

Ejecutar esta auditoría **30 minutos antes** de pasar al frente con el comité evaluador:

| Componente | Verificación | Comando de Verificación |
| :--- | :--- | :--- |
| **VM en GCP** | Instancia `RUNNING` en Google Cloud Console | `gcloud compute instances list` o panel web |
| **Servicio Systemd** | Servicio activo y corriendo 24/7 sin fallos | `sudo systemctl status tradingbot` |
| **API Keys** | `GEMINI_API_KEY` y `TELEGRAM_BOT_TOKEN` válidos | Probar `/status` y `/signal BTCUSDT` en Telegram |
| **Dashboard Web** | URL pública activa y accesible desde el navegador del jurado | Abrir `http://<IP_PUBLICA_GCP>:8080/` |
| **Logs en Vivo** | Sin errores de Python en `trading_bot.log` | `tail -n 20 /opt/trading-bot-pro/trading_bot.log` |

---

### 📱 2. SECUENCIA EXACTA PARA LA DEMOSTRACIÓN EN TELEGRAM

Durante la defensa, proyectar la pantalla de Telegram o el teléfono en vivo y ejecutar en orden los siguientes comandos ante el profesor:

1. **Demostración de Conectividad con la VM de GCP:**
   - Enviar: `/status`
   - *Respuesta esperada:* Muestra CPU %, RAM en MB, Disco libre y confirmación del servicio activo.
   - *Discurso:* *"Como pueden observar, el bot está corriendo en segundo plano en nuestra VM de Google Cloud Compute Engine e2-medium con persistencia systemd."*

2. **Demostración de Machine Learning y Análisis Cuantitativo:**
   - Enviar: `/signal BTCUSDT`
   - *Respuesta esperada:* Muestra decisión de Random Forest (`BUY`/`SELL`/`HOLD`), certeza %, RSI, MACD, SAR Parabólico, pendiente de regresión y el análisis institucional del Agente Gemini Enterprise.
   - *Discurso:* *"Aquí el modelo Random Forest evalúa la confluencia entre los osciladores y la tendencia del SAR y la regresión lineal, respaldado por la síntesis del Agente Gemini."*

3. **Demostración de Simulación Monte Carlo:**
   - Enviar: `/montecarlo BTCUSDT`
   - *Respuesta esperada:* Reporte estocástico con probabilidad de Take Profit $P(\text{TP})$, probabilidad de Stop Loss $P(\text{SL})$, VaR al 95% y Edge Ratio.
   - *Discurso:* *"Antes de ejecutar la orden, computamos 1,000 caminos Brownianos para verificar asimetría estadística en el trade."*

4. **Demostración de Ejecución de Orden en Vivo:**
   - Enviar: `/buy BTCUSDT`
   - *Respuesta esperada:* Notificación de orden de compra ejecutada en la VM con Stop Loss dinámico y Take Profit basados en ATR.
   - Enviar: `/positions` para mostrar la posición abierta con PnL flotante.
   - Enviar: `/balance` para mostrar el capital total y la ganancia acumulada.

---

### 🖥️ 3. DEMOSTRACIÓN DEL DASHBOARD WEB Y TRADINGVIEW

1. **Ingreso y Autenticación:**
   - Abrir `http://<IP_PUBLICA_GCP>:8080/login`
   - Ingresar con credenciales: `admin` / `cuantitativo2026`.
2. **Panel de Métricas (KPIs):**
   - Señalar el **Profit Factor** ($> 1.5$), el **Win Rate**, la **Ganancia Acumulada** y el **Sharpe Ratio**.
3. **TradingView Widget:**
   - Mostrar cómo se actualizan las velas en tiempo real y cómo se puede cambiar de temporalidad o de ticker usando el buscador.
4. **Demostración de Sentimiento NLP (Gemini Enterprise):**
   - Mostrar la tarjeta de **Sentimiento NLP & Noticias** en la columna derecha.
   - Explicar la barra de sentimiento (-1.0 a +1.0) y la síntesis semántica: *"Aquí Gemini Enterprise procesa los titulares financieros en tiempo real y asigna un score NLP; si detecta noticias adversas de alto impacto, activa una regla de veto para proteger la cartera."*
5. **Pestaña de Machine Learning & Diebold-Mariano:**
   - Mostrar la tabla comparativa de simulación de hiperparámetros (RMSE, MAE, SMAPE, MAPE, Variance Score $R^2$).
   - Explicar la Matriz de Confusión.
   - **Punto Clave:** Explicar el **Test de Diebold-Mariano**: *"Calculamos el estadístico DM con corrección de Harvey-Leybourne-Newbold para demostrar con un p-valor menor a 0.05 que nuestro Random Forest supera significativamente al benchmark."*
6. **Pestaña de Configuración Personalizada por Usuario:**
   - Abrir la pestaña **Configuración de Métricas**.
   - Demostrar cómo cualquier trader o profesor puede activar/desactivar indicadores específicos (RSI, SAR, MACD, etc.) o ajustar el riesgo ATR y la certeza mínima de Machine Learning, guardando sus preferencias al instante.
7. **Pestaña Monte Carlo:**
   - Mostrar el abanico de percentiles (p5 a p95) y las líneas horizontales de Stop Loss y Take Profit.

---

### 🛡️ 4. COMANDOS DE RESCATE (SI ALGO PARECE CONGELARSE)

Si por alguna razón la VM se detiene antes de tu turno:
```bash
# Conectarse a la VM vía SSH de GCP:
ssh -i ~/.ssh/google_compute_engine ubuntu@<IP_PUBLICA_GCP>

# Reiniciar el servicio al instante:
sudo systemctl restart tradingbot

# Ver logs inmediatos:
tail -f /opt/trading-bot-pro/trading_bot.log
```
Con este procedimiento, la demostración es 100% infalible y asegura la máxima puntuación en todos los rubros.

