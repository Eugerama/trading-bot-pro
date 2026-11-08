# INFORME TÉCNICO Y ARQUITECTURA DEL SISTEMA
## Proyecto Integrador: Bot de Trading Cuantitativo Autónomo en GCP Compute Engine + Telegram ChatOps + Machine Learning & Validación Diebold-Mariano

---

### 1. Resumen Ejecutivo
El presente proyecto integrador implementa un sistema autónomo e institucional de trading cuantitativo desplegado en la nube de **Google Cloud Platform (Compute Engine, Serie E2)**, alimentado por modelos de **Machine Learning (Random Forest y KNN)**, indicadores econométricos avanzados (Medias Móviles de Regresión, MACD, RSI, SAR Parabólico, Oscilador Estocástico y ATR), simulaciones estocásticas **Monte Carlo (Movimiento Browniano Geométrico)** y asistencia explicativa mediante **Google Gemini Enterprise (Vertex AI / AI Studio)**.

La validación predictiva se realiza bajo rigor estadístico mediante la prueba formal de **Diebold-Mariano (1995)** con corrección de **Harvey, Leybourne y Newbold (HLN, 1997)**, junto a la simulación comparativa de hiperparámetros sobre métricas clave: **RMSE, MAE, SMAPE, MAPE, Variance Score ($R^2$), Win Rate y Profit Factor**.

---

### 2. Formulación y Fundamentos Matemáticos

#### 2.1 Test de Validación Estadística Diebold-Mariano (Diebold & Mariano, 1995)
Para evaluar si la superioridad predictiva del modelo Random Forest ($M_1$) sobre el modelo de referencia KNN ($M_2$) es estadísticamente significativa y no producto del azar, se implementa el test de Diebold-Mariano sobre una muestra de tamaño $T$ con horizonte de pronóstico $h$:

1. **Diferencial de Pérdida ($d_t$):**
   $$d_t = L(e_{1,t}) - L(e_{2,t})$$
   donde bajo pérdida cuadrática (MSE loss):
   $$d_t = (y_t - \hat{y}_{1,t})^2 - (y_t - \hat{y}_{2,t})^2$$

2. **Media de la Diferencial de Pérdida:**
   $$\bar{d} = \frac{1}{T} \sum_{t=1}^{T} d_t$$

3. **Varianza Asintótica con Corrección de Autocorrelación (Newey-West / Bartlett Kernel):**
   $$\hat{V}(\bar{d}) = \frac{1}{T} \left( \hat{\gamma}_0 + 2 \sum_{k=1}^{h-1} \left(1 - \frac{k}{h}\right) \hat{\gamma}_k \right)$$
   donde $\hat{\gamma}_k$ es la autocovarianza de orden $k$:
   $$\hat{\gamma}_k = \frac{1}{T} \sum_{t=k+1}^{T} (d_t - \bar{d})(d_{t-k} - \bar{d})$$

4. **Corrección de Harvey, Leybourne y Newbold (HLN, 1997) para Muestras Pequeñas:**
   $$DM^* = \left( \frac{\bar{d}}{\sqrt{\hat{V}(\bar{d})}} \right) \times \sqrt{\frac{T + 1 - 2h + h(h-1)/T}{T}}$$

5. **Prueba de Hipótesis:**
   - $H_0: E[d_t] = 0$ (Ambos modelos tienen la misma precisión predictiva).
   - $H_1: E[d_t] \neq 0$ (Existe diferencia estadísticamente significativa).
   Se rechaza $H_0$ si $|DM^*| > t_{\alpha/2, T-1}$ (o $p\text{-valor} < 0.05$). Un $DM^* < 0$ significativo demuestra que el Random Forest genera un error menor que el benchmark.

---

#### 2.2 Simulación Monte Carlo de Escenarios (Movimiento Browniano Geométrico - GBM)
La dinámica de precios de los activos se modela mediante la Ecuación Diferencial Estocástica (SDE):
$$dS_t = \mu S_t dt + \sigma S_t dW_t$$

Por el **Lema de Itô**, la solución exacta en tiempo discreto es:
$$S_{t+\Delta t} = S_t \exp\left( \left( \mu - \frac{1}{2}\sigma^2 \right)\Delta t + \sigma \sqrt{\Delta t} Z \right), \quad Z \sim \mathcal{N}(0, 1)$$

Se simulan $N = 1,000$ trayectorias sobre un horizonte de $H = 48$ periodos para calcular:
- **Probabilidad de Take Profit:** $P(\tau_{\text{TP}} < \tau_{\text{SL}})$
- **Value at Risk al 95% ($VaR_{95\%}$):**
  $$VaR_{0.05} = \inf \{ l \in \mathbb{R} : P(L > l) \le 0.05 \}$$
- **Conditional Value at Risk ($CVaR_{95\%}$ / Expected Shortfall):**
  $$CVaR_{0.05} = E[ L \mid L \ge VaR_{0.05} ]$$

---

#### 2.3 Medias Móviles de Regresión Lineal (Ordinary Least Squares Deslizante)
Para cada ventana temporal de longitud $N$, se ajusta la recta:
$$\hat{y}_i = \hat{m} x_i + \hat{b}$$
$$\hat{m} = \frac{\sum_{i=1}^N (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^N (x_i - \bar{x})^2}, \quad \hat{b} = \bar{y} - \hat{m} \bar{x}$$
El valor de la curva en el periodo actual es $\hat{y}_N = \hat{m}(N-1) + \hat{b}$, y su pendiente $\hat{m}$ actúa como filtro de régimen tendencial.

---

#### 2.4 SAR Parabólico (Parabolic Stop and Reverse)
El SAR evoluciona incrementalmente según:
$$SAR_{t} = SAR_{t-1} + AF \times (EP - SAR_{t-1})$$
donde $AF$ (Acceleration Factor) inicia en $0.02$ y se incrementa en pasos de $0.02$ hasta un techo de $0.20$ conforme el precio marca nuevos extremos ($EP$).

---

#### 2.5 Gestión de Riesgo Dinámica con ATR (Average True Range)
El True Range se define como:
$$TR_t = \max(H_t - L_t, |H_t - C_{t-1}|, |L_t - C_{t-1}|)$$
$$ATR_t = \text{WilderSmoothing}(TR_t, 14)$$

- **Stop Loss Dinámico:** $SL = P_{\text{entry}} \mp (1.5 \times ATR)$
- **Take Profit Dinámico:** $TP = P_{\text{entry}} \pm (2.5 \times ATR)$
- **Dimensionamiento de Lote:**
  $$\text{Units} = \frac{\text{Capital} \times \text{Risk}\%}{|P_{\text{entry}} - SL|}$$

---

#### 2.6 Procesamiento de Lenguaje Natural (NLP) y Veto Institucional por Sentimiento (Gemini Enterprise)
Para cumplir con el requerimiento de análisis no estructurado, se implementa un modelo de **Procesamiento de Lenguaje Natural (NLP)** asistido por **Google Gemini Enterprise (Vertex AI)**:
1. **Extracción y Tokenización Semántica:** Se extraen titulares de noticias macroeconómicas e institucionales del activo en tiempo real.
2. **Puntuación de Sentimiento ($S_{\text{NLP}} \in [-1.0, +1.0]$):**
   $$S_{\text{NLP}} = f_{\text{Gemini}}(\text{Titulares}, \text{Contexto Macroeconómico})$$
   - $S_{\text{NLP}} \ge +0.50$: Clasificación *MUY ALCISTA* (Fuerte flujo comprador institucional).
   - $-0.15 \le S_{\text{NLP}} < +0.50$: Clasificación *ALCISTA / NEUTRAL*.
   - $S_{\text{NLP}} \le -0.50$: Clasificación *MUY BAJISTA* (Pánico / Fuga de liquidez).
3. **Mecanismo de Veto Institucional:**
   $$\text{Decisión Final} = \begin{cases} 
   \text{HOLD}, & \text{si } \text{Señal}_{\text{RF}} = \text{BUY} \text{ y } S_{\text{NLP}} \le -0.50 \quad \text{(Veto por Pánico)} \\
   \text{HOLD}, & \text{si } \text{Señal}_{\text{RF}} = \text{SELL} \text{ y } S_{\text{NLP}} \ge +0.50 \quad \text{(Veto por Euforia)} \\
   \text{Señal}_{\text{RF}}, & \text{en caso de confluencia técnica y fundamental}
   \end{cases}$$

---

### 3. Métricas Cuantitativas Exigidas

| Métrica | Fórmula Matemática | Propósito |
| :--- | :--- | :--- |
| **RMSE** | $\sqrt{\frac{1}{n} \sum (y - \hat{y})^2}$ | Penaliza fuertemente grandes desviaciones |
| **MAE** | $\frac{1}{n} \sum \|y - \hat{y}\|$ | Error absoluto medio sin sesgo cuadrático |
| **SMAPE** | $\frac{100\%}{n} \sum \frac{\|y - \hat{y}\|}{(\|y\| + \|\hat{y}\|)/2}$ | Error porcentual simétrico acotado entre 0% y 200% |
| **MAPE** | $\frac{100\%}{n} \sum \|\frac{y - \hat{y}}{y}\|$ | Error porcentual estándar |
| **Variance Score ($R^2$)** | $1 - \frac{\sum (y - \hat{y})^2}{\sum (y - \bar{y})^2}$ | Porcentaje de varianza explicada por el modelo |
| **Profit Factor** | $\frac{\sum \text{Ganancias Brutas}}{\sum \|\text{Pérdidas Brutas}\|}$ | Asimetría de rentabilidad (Objetivo: $> 1.5$) |
| **Win Rate** | $\frac{\text{Trades Ganadores}}{\text{Trades Totales}} \times 100\%$ | Tasa de acierto de las posiciones ejecutadas |

---

### 4. Arquitectura de Software y Despliegue en GCP Compute Engine

```
[ Mercado en Vivo: yfinance / Binance / Bybit ] ──► [ Noticias Financieras / RSS ]
                      │                                       │
                      ▼                                       ▼
            [ core/data_feed.py ]                   [ core/nlp_sentiment.py ]
                      │                             (Gemini Enterprise NLP)
                      ▼                                       │
            [ core/indicators.py ] ──► (Regresión, SAR, ATR)  │
                      │                                       │
                      ▼                                       │
            [ core/ml_engine.py ] ──► (Random Forest)         │
                      │                                       │
         ┌────────────┴────────────┐                          │
         ▼                         ▼                          │
[ core/monte_carlo.py ]   [ core/risk_manager.py ]            │
(1,000 trayectorias GBM)  (SL/TP dinámico y Sizing)           │
         │                         │                          │
         └────────────┬────────────┘                          │
                      ▼                                       │
            [ core/strategy.py ] ◄────────────────────────────┘
            (Confluencia Técnica + ML + Filtro de Veto NLP)
                      │
         ┌────────────┴────────────┐
         ▼                         ▼
[ core/gemini_agent.py ]   [ core/execution_engine.py ]
(Dictamen Institucional)   (Órdenes, Balance, PnL)
         │                         │
         ├─────────────────────────┤
         ▼                         ▼
[ telegram_bot/bot.py ]    [ web/server.py (FastAPI) ]
  (ChatOps en Vivo 24/7)    (TradingView Instantáneo + Settings UI)
```

- **Infraestructura Cloud:** Máquina Virtual Google Cloud Compute Engine (e2-medium, 2 vCPU, 4 GB RAM, Ubuntu 22.04 LTS).
- **Persistencia 24/7:** Servicio `systemd` (`tradingbot.service`) con reinicio automático (`Restart=always`).
- **Seguridad:** Variables de entorno desacopladas (`.env`), puerto 8080 en cortafuegos de GCP y autenticación de sesión para el Dashboard.

