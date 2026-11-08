# 🚀 Quantitative Trading Bot & AI Dashboard

[![Google Cloud Platform](https://img.shields.io/badge/GCP-Compute%20Engine%20(e2--medium)-4285F4?logo=google-cloud&logoColor=white)](https://cloud.google.com/)
[![Gemini Enterprise](https://img.shields.io/badge/AI-Gemini%20Enterprise%20(Vertex%20AI)-1a73e8?logo=google&logoColor=white)](https://aistudio.google.com/)
[![Telegram ChatOps](https://img.shields.io/badge/ChatOps-Telegram%20Bot%20API-2CA5E0?logo=telegram&logoColor=white)](https://telegram.org/)
[![Machine Learning](https://img.shields.io/badge/ML-Random%20Forest%20%7C%20Diebold--Mariano-FF6F00?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![TradingView](https://img.shields.io/badge/Charts-TradingView%20Widget-2962FF?logo=tradingview&logoColor=white)](https://tradingview.com/)

Sistema institucional de trading cuantitativo algorítmico, simulación estocástica Monte Carlo, modelos de Machine Learning (Random Forest) con validación formal mediante el **Test de Diebold-Mariano (1995 / HLN 1997)**, ChatOps bidireccional vía **Telegram Bot API**, y Dashboard interactivo con **TradingView en tiempo real** y autenticación de usuarios. Desplegable en **Google Cloud Compute Engine** con soporte 24/7 mediante `systemd`.

---

## 📑 Tabla de Contenidos
- [Características Principales](#-características-principales)
- [Arquitectura del Proyecto](#-arquitectura-del-proyecto)
- [Estrategia e Indicadores Cuantitativos](#-estrategia-e-indicadores-cuantitativos)
- [Validación Rigurosa de Machine Learning](#-validación-rigurosa-de-machine-learning)
- [Instalación y Configuración](#-instalación-y-configuración)
- [Despliegue en Google Cloud Compute Engine](#-despliegue-en-google-cloud-compute-engine)
- [Comandos ChatOps en Telegram](#-comandos-chatops-en-telegram)
- [Entregables Obligatorios](#-entregables-obligatorios)

---

## 🌟 Características Principales
1. **Modelos de Machine Learning:**
   - Clasificador **Random Forest** optimizado con TimeSeriesSplit.
   - Benchmark competitivo con **K-Nearest Neighbors (KNN)**.
   - **Simulación de Hiperparámetros:** Evaluación de RMSE, MAE, SMAPE, MAPE, Variance Score ($R^2$), Accuracy, F1-Score y Matriz de Confusión.
   - **Test de Validación Diebold-Mariano (1995):** Corrección HLN (1997) para certificar matemáticamente la superioridad predictiva del modelo ($p < 0.05$).
2. **Simulación Estocástica Monte Carlo:**
   - 1,000 caminos simulados mediante Movimiento Browniano Geométrico (GBM) y Lema de Itô.
   - Cálculo en tiempo real de $P(\text{Take Profit})$, $P(\text{Stop Loss})$, Value at Risk ($VaR_{95\%}$) y Expected Shortfall ($CVaR_{95\%}$).
3. **Indicadores Técnicos Cuantitativos:**
   - Medias móviles de regresión lineal (Slope & Curve).
   - SAR Parabólico (Parabolic Stop and Reverse).
   - MACD (Moving Average Convergence Divergence) e Histograma.
   - RSI (Relative Strength Index de Wilder).
   - Oscilador Estocástico (%K, %D) y Bandas de Bollinger.
   - ATR (Average True Range) para Stop Loss dinámico y dimensionamiento de posición.
4. **Dashboard Web Interactivo con TradingView:**
   - Gráfico oficial TradingView en tiempo real con selector dinámico de tickers.
   - Autenticación de usuarios multiusuario (Login con roles).
   - Panel de Posiciones Activas y Órdenes ejecutadas.
   - Buscador universal de criptomonedas y acciones (BTCUSDT, ETHUSDT, SOLUSDT, NVDA, AAPL, SPY, etc.).
   - Visualizador de métricas: **Win Rate, Profit Factor, Ganancia Acumulada, Sharpe Ratio y Max Drawdown**.
   - Asistente de IA cuantitativo impulsado por **Google Gemini Enterprise**.
5. **ChatOps en Vivo con Telegram:**
   - Monitoreo en tiempo real de la VM de GCP (`/status`).
   - Envío y recepción de señales y órdenes desde Telegram (`/signal`, `/buy`, `/sell`, `/balance`, `/montecarlo`).

---

## 📁 Arquitectura del Proyecto

```
trading-bot-pro/
├── core/
│   ├── indicators.py        # Medias de regresión, MACD, RSI, SAR, ATR, Estocástico
│   ├── data_feed.py         # Ingesta con yfinance y generador browniano de contingencia
│   ├── ml_engine.py         # Random Forest, Grid de Hiperparámetros, Diebold-Mariano Test
│   ├── monte_carlo.py       # Simulación Monte Carlo (GBM, VaR, CVaR, P(TP))
│   ├── risk_manager.py      # Stop Loss / Take Profit dinámico con ATR y Circuit Breaker
│   ├── strategy.py          # Estrategia híbrida (indicadores + ML + umbrales)
│   ├── gemini_agent.py      # Agente Cuantitativo con Google Gemini Enterprise
│   └── execution_engine.py  # Motor de órdenes, balance, PnL y persistencia JSON
├── telegram_bot/
│   └── bot.py               # ChatOps 24/7 con Telegram Bot API
├── web/
│   ├── server.py            # Servidor FastAPI y API REST
│   └── templates/
│       ├── login.html       # Pantalla de autenticación institucional
│       └── index.html       # Dashboard interactivo con TradingView y Chart.js
├── gcp/
│   ├── setup_vm.sh          # Script de aprovisionamiento en Ubuntu 22.04 LTS
│   └── tradingbot.service   # Servicio systemd de persistencia continua 24/7
├── notebooks/
│   └── 01_model_training_and_diebold_mariano.ipynb # Notebook reproducible
├── docs/
│   ├── INFORME_TECNICO_PROYECTO.md  # Informe formal con ecuaciones y tablas
│   └── GUIA_DEFENSA_EN_VIVO.md      # Protocolo anti-penalizaciones
├── tests/
│   └── test_quantitative_system.py # Pruebas unitarias de integridad
├── .env.example             # Plantilla de variables de entorno segura
├── requirements.txt         # Dependencias del proyecto
├── Dockerfile               # Contenedor para Google Cloud Run / Docker
├── docker-compose.yml       # Orquestación multicontenedor
└── main.py                  # Orquestador del servidor web, bot y trading loop
```

---

## ⚡ Instalación y Configuración Local

1. **Clonar el repositorio:**
   ```bash
   git clone https://github.com/tu-usuario/trading-bot-pro.git
   cd trading-bot-pro
   ```

2. **Crear y activar entorno virtual:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # En Linux/Mac
   ```

3. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configurar variables de entorno:**
   ```bash
   cp .env.example .env
   # Editar .env con tus claves de Gemini API y Telegram Bot Token
   ```

5. **Iniciar el sistema completo:**
   ```bash
   python main.py
   ```
   Abre tu navegador en `http://localhost:8080` (Usuario: `admin` | Clave: `cuantitativo2026`).

---

## ☁️ Despliegue en Google Cloud Compute Engine

1. Crear una instancia en Google Cloud Console:
   - Serie: `E2` | Tipo: `e2-medium` (2 vCPU, 4 GB RAM)
   - Sistema Operativo: `Ubuntu 22.04 LTS`
   - Disco: 25 GB Persistente Estándar
   - Cortafuegos: Marcar "Permitir tráfico HTTP" y "Permitir tráfico HTTPS".
2. Conectarse vía SSH a la máquina virtual:
   ```bash
   git clone https://github.com/tu-usuario/trading-bot-pro.git
   cd trading-bot-pro
   cp .env.example .env  # Configurar tus credenciales
   chmod +x gcp/setup_vm.sh
   ./gcp/setup_vm.sh
   ```
3. El servicio se instalará bajo `systemd` y correrá de forma ininterrumpida las 24 horas del día, 7 días a la semana.

---

## 🤖 Comandos ChatOps en Telegram

| Comando | Acción |
| :--- | :--- |
| `/status` | Reporta el uso de CPU, RAM, disco de la VM y estado del servicio |
| `/signal <TICKER>` | Ejecuta Random Forest + Indicadores + Gemini en tiempo real |
| `/montecarlo <TICKER>` | Simula 1,000 caminos Brownianos y reporta $P(\text{TP})$, $P(\text{SL})$ y VaR |
| `/balance` | Muestra el capital total, equidad, margen y ganancia acumulada |
| `/positions` | Detalla las posiciones activas en la VM con Stop Loss y Take Profit |
| `/metrics` | Informa el Win Rate, Profit Factor y Sharpe Ratio de la cartera |
| `/buy <TICKER>` | Ejecuta una orden de compra manual con dimensionamiento ATR |
| `/sell <TICKER>` | Ejecuta una orden de venta manual |
| `/close <TICKER>` | Cierra manualmente una posición abierta |

---

## 🎓 Cumplimiento de Rúbrica de Evaluación

- **Simulación de Hiperparámetros:** Tabla comparativa completa con RMSE, MAE, SMAPE, MAPE y Variance Score ($R^2$).
- **Test de Diebold-Mariano:** Implementación matemática con corrección de Harvey-Leybourne-Newbold (HLN).
- **Prevención de Penalizaciones:** Auditoría previa de API Keys, Fail-safe estocástico para conectividad garantizada, servicio `systemd` a prueba de reinicios.

