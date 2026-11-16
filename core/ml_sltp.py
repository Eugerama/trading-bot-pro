"""
Módulo de Stop Loss / Take Profit Dinámico con Machine Learning.

En lugar de multiplicadores ATR fijos (ej. 1.5x SL / 2.5x TP), este módulo aprende,
por activo, cuánto se mueve realmente el precio EN CONTRA (MAE) y A FAVOR (MFE) de una
posición en las próximas N velas, dadas las condiciones técnicas actuales.

Metodología:
- Targets (normalizados por ATR para que sean comparables entre activos/regímenes):
    * LONG:  MFE = (max(high[t+1..t+H]) - close_t) / ATR_t
             MAE = (close_t - min(low[t+1..t+H])) / ATR_t
    * SHORT: simétrico (MFE usa los mínimos, MAE los máximos)
- Modelos: Gradient Boosting con pérdida cuantílica (pinball loss):
    * Stop Loss  = cuantil q_sl (por defecto 0.75) del MAE esperado.
      El stop se coloca más allá del 75% de los retrocesos "normales" -> evita stops por ruido.
    * Take Profit = cuantil q_tp (por defecto 0.50) del MFE esperado.
      Objetivo alcanzable en la mediana de los escenarios.
- Validación out-of-sample (último 25% temporal): cobertura empírica del cuantil y
  pinball loss del modelo vs. el baseline de multiplicadores ATR fijos.
- Compuerta de validación: el ML sólo se usa para un activo/lado si en test out-of-sample
  supera al baseline ATR fijo en pinball loss tanto de SL como de TP. Si no, se usa ATR fijo.
- Re-entrenamiento walk-forward cada 6 horas por activo, con caché en disco.
- Si no hay scikit-learn, datos suficientes, modelo validado, el RiskManager vuelve al método ATR fijo.
"""

import os
import math
import time
import pickle
from datetime import datetime
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import HistGradientBoostingRegressor
    SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover - depende del entorno
    HistGradientBoostingRegressor = None
    SKLEARN_AVAILABLE = False

from core.ml_engine import FEATURE_COLUMNS, build_feature_frame


def compute_excursion_targets(df_feat: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    Calcula MFE/MAE futuros (en múltiplos de ATR) para posiciones LONG y SHORT.
    Las últimas `horizon` filas quedan en NaN (no tienen futuro completo).
    """
    high = df_feat['high'].to_numpy(dtype=float)
    low = df_feat['low'].to_numpy(dtype=float)
    close = df_feat['close'].to_numpy(dtype=float)
    atr = df_feat['atr_14'].to_numpy(dtype=float)
    n = len(df_feat)

    fut_max = np.full(n, np.nan)
    fut_min = np.full(n, np.nan)
    if n > horizon:
        # Ventana deslizante sobre las velas t+1 .. t+H
        hw = np.lib.stride_tricks.sliding_window_view(high[1:], horizon)
        lw = np.lib.stride_tricks.sliding_window_view(low[1:], horizon)
        fut_max[:len(hw)] = hw.max(axis=1)
        fut_min[:len(lw)] = lw.min(axis=1)

    safe_atr = np.where(atr > 0, np.maximum(atr, close * 0.005), np.nan)

    return pd.DataFrame({
        "long_mfe": (fut_max - close) / safe_atr,
        "long_mae": (close - fut_min) / safe_atr,
        "short_mfe": (close - fut_min) / safe_atr,
        "short_mae": (fut_max - close) / safe_atr,
    }, index=df_feat.index).clip(lower=0.0)


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    """Pérdida cuantílica (pinball). Menor es mejor."""
    diff = y_true - y_pred
    return float(np.mean(np.maximum(q * diff, (q - 1.0) * diff)))


class MLStopLossTakeProfit:
    """Predice multiplicadores ATR dinámicos de SL/TP por activo y por lado (BUY/SELL)."""

    CACHE_FILE = "ml_sltp_cache.pkl"
    RETRAIN_INTERVAL_SEC = 6 * 3600
    MIN_SAMPLES = 300

    def __init__(
        self,
        horizon_bars: int = 24,
        sl_quantile: float = 0.75,
        tp_quantile: float = 0.50,
        sl_bounds: tuple = (0.8, 4.0),
        tp_bounds: tuple = (1.0, 6.0),
        min_risk_reward: float = 1.2,
        cache_file: Optional[str] = None,
    ):
        self.horizon_bars = horizon_bars
        self.sl_quantile = sl_quantile
        self.tp_quantile = tp_quantile
        self.sl_bounds = sl_bounds
        self.tp_bounds = tp_bounds
        self.min_risk_reward = min_risk_reward
        self.cache_file = cache_file if cache_file is not None else self.CACHE_FILE

        # models_by_key[ticker] = {"BUY": {"sl": model, "tp": model}, "SELL": {...}}
        self.models_by_key: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self.trained_at_by_key: Dict[str, float] = {}
        self.metrics_by_key: Dict[str, Dict[str, Any]] = {}
        self._load_cache()

    # ------------------------------------------------------------------
    # Persistencia
    # ------------------------------------------------------------------
    def _load_cache(self):
        if not self.cache_file or not os.path.exists(self.cache_file):
            return
        try:
            # Archivo local generado por el propio bot (mismo patrón que ml_engine). No cargar pickles de terceros.
            with open(self.cache_file, "rb") as f:
                data = pickle.load(f)
            self.models_by_key = data.get("models_by_key", {})
            self.trained_at_by_key = data.get("trained_at_by_key", {})
            self.metrics_by_key = data.get("metrics_by_key", {})
        except Exception as e:
            print(f"[ML-SLTP] Advertencia al cargar caché: {e}")

    def _save_cache(self):
        if not self.cache_file:
            return
        try:
            with open(self.cache_file, "wb") as f:
                pickle.dump({
                    "models_by_key": self.models_by_key,
                    "trained_at_by_key": self.trained_at_by_key,
                    "metrics_by_key": self.metrics_by_key,
                }, f)
        except Exception as e:
            print(f"[ML-SLTP] Error guardando caché: {e}")

    # ------------------------------------------------------------------
    # Entrenamiento
    # ------------------------------------------------------------------
    def _new_model(self, quantile: float):
        return HistGradientBoostingRegressor(
            loss="quantile",
            quantile=quantile,
            max_iter=150,
            max_depth=4,
            learning_rate=0.05,
            min_samples_leaf=20,
            random_state=42,
        )

    def train(self, df_raw: pd.DataFrame, key: str) -> Dict[str, Any]:
        """Entrena los 4 modelos cuantílicos (SL/TP x BUY/SELL) para un activo."""
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn no disponible para ML-SLTP.")

        feat = build_feature_frame(df_raw)
        targets = compute_excursion_targets(feat, self.horizon_bars)
        data = pd.concat([feat[FEATURE_COLUMNS], targets], axis=1).dropna()

        if len(data) < self.MIN_SAMPLES:
            raise ValueError(f"Datos insuficientes para ML-SLTP ({len(data)} muestras < {self.MIN_SAMPLES}).")

        X = data[FEATURE_COLUMNS].to_numpy()
        split = int(len(data) * 0.75)
        # Embargo: las etiquetas de las últimas `horizon` filas de train ven precios del periodo de test
        train_end = max(1, split - self.horizon_bars)

        models: Dict[str, Dict[str, Any]] = {}
        metrics: Dict[str, Any] = {
            "samples": int(len(data)),
            "horizon_bars": self.horizon_bars,
            "sl_quantile": self.sl_quantile,
            "tp_quantile": self.tp_quantile,
        }

        for side, prefix in (("BUY", "long"), ("SELL", "short")):
            y_mae = data[f"{prefix}_mae"].to_numpy()
            y_mfe = data[f"{prefix}_mfe"].to_numpy()

            # 1) Validación out-of-sample
            sl_eval = self._new_model(self.sl_quantile).fit(X[:train_end], y_mae[:train_end])
            tp_eval = self._new_model(self.tp_quantile).fit(X[:train_end], y_mfe[:train_end])
            sl_pred = sl_eval.predict(X[split:])
            tp_pred = tp_eval.predict(X[split:])
            y_mae_te, y_mfe_te = y_mae[split:], y_mfe[split:]

            # Baseline: multiplicadores ATR fijos clásicos (1.5 / 2.5)
            base_sl = np.full_like(y_mae_te, 1.5)
            base_tp = np.full_like(y_mfe_te, 2.5)

            sl_model_loss = pinball_loss(y_mae_te, sl_pred, self.sl_quantile)
            sl_fixed_loss = pinball_loss(y_mae_te, base_sl, self.sl_quantile)
            tp_model_loss = pinball_loss(y_mfe_te, tp_pred, self.tp_quantile)
            tp_fixed_loss = pinball_loss(y_mfe_te, base_tp, self.tp_quantile)

            metrics[side] = {
                # Fracción de trades cuyo retroceso real NO tocó el SL predicho (ideal ≈ q_sl)
                "sl_coverage": round(float(np.mean(y_mae_te <= sl_pred)), 4),
                # Fracción de trades que SÍ alcanzaron el TP predicho (ideal ≈ 1 - q_tp)
                "tp_hit_rate": round(float(np.mean(y_mfe_te >= tp_pred)), 4),
                "sl_pinball_model": round(sl_model_loss, 5),
                "sl_pinball_fixed_atr": round(sl_fixed_loss, 5),
                "tp_pinball_model": round(tp_model_loss, 5),
                "tp_pinball_fixed_atr": round(tp_fixed_loss, 5),
                "test_size": int(len(y_mae_te)),
                # Sólo se usa el ML si mejora al ATR fijo fuera de muestra en SL y en TP
                "validated": bool(sl_model_loss < sl_fixed_loss and tp_model_loss < tp_fixed_loss),
            }

            # 2) Modelo final con todos los datos (walk-forward)
            models[side] = {
                "sl": self._new_model(self.sl_quantile).fit(X, y_mae),
                "tp": self._new_model(self.tp_quantile).fit(X, y_mfe),
            }

        metrics["trained_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.models_by_key[key] = models
        self.trained_at_by_key[key] = time.time()
        self.metrics_by_key[key] = metrics
        self._save_cache()
        return metrics

    def _ensure_trained(self, df_raw: pd.DataFrame, key: str) -> bool:
        age = time.time() - self.trained_at_by_key.get(key, 0)
        if key in self.models_by_key and age <= self.RETRAIN_INTERVAL_SEC:
            return True
        try:
            self.train(df_raw, key)
            return True
        except Exception as e:
            print(f"[ML-SLTP] No se pudo entrenar {key}: {e}")
            # Si hay un modelo previo (aunque viejo) se sigue usando
            return key in self.models_by_key

    # ------------------------------------------------------------------
    # Predicción
    # ------------------------------------------------------------------
    def predict_multipliers(self, df_raw: pd.DataFrame, side: str, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Retorna {"sl_mult", "tp_mult", "raw_sl_mult", "raw_tp_mult"} en múltiplos de ATR,
        o None si el modelo no está disponible (el llamador debe usar el método ATR fijo).
        """
        side = (side or "").upper()
        if side not in ("BUY", "SELL") or not SKLEARN_AVAILABLE or df_raw is None:
            return None

        key = (ticker or "").upper().strip() or "DEFAULT"
        if not self._ensure_trained(df_raw, key):
            return None

        side_metrics = self.metrics_by_key.get(key, {}).get(side, {})
        if not side_metrics.get("validated", False):
            return None  # El ML no superó al ATR fijo en validación -> fallback

        try:
            feat = build_feature_frame(df_raw)
            row = feat[FEATURE_COLUMNS].iloc[-1:]
            if row.isna().any(axis=None):
                return None
            x = row.to_numpy()
            side_models = self.models_by_key[key][side]
            raw_sl = float(side_models["sl"].predict(x)[0])
            raw_tp = float(side_models["tp"].predict(x)[0])
        except Exception as e:
            print(f"[ML-SLTP] Error de predicción para {key}: {e}")
            return None

        sl_mult = round(float(np.clip(raw_sl, *self.sl_bounds)), 3)
        tp_mult = round(float(np.clip(raw_tp, *self.tp_bounds)), 3)
        # Garantizar una relación Riesgo/Beneficio mínima (redondeo hacia arriba para no quedar por debajo)
        tp_mult = max(tp_mult, math.ceil(sl_mult * self.min_risk_reward * 1000) / 1000)

        return {
            "sl_mult": sl_mult,
            "tp_mult": tp_mult,
            "raw_sl_mult": round(raw_sl, 3),
            "raw_tp_mult": round(raw_tp, 3),
            # True si el ML espera menos recorrido a favor que el R/B mínimo exige
            # (el TP se forzó hacia arriba -> setup con poca ventaja estadística)
            "low_edge": bool(raw_tp < max(raw_sl, self.sl_bounds[0]) * self.min_risk_reward),
        }

    def get_metrics(self, ticker: Optional[str] = None) -> Dict[str, Any]:
        if ticker:
            return self.metrics_by_key.get(ticker.upper().strip(), {})
        return dict(self.metrics_by_key)
