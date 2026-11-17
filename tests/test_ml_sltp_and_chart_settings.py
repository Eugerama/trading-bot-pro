"""
Pruebas para:
1. SL/TP dinámico con Machine Learning (core/ml_sltp.py + RiskManager).
2. Configuración compartida de la gráfica (core/chart_settings.py) usada por Telegram y el Dashboard.
"""

import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from core.data_feed import generate_synthetic_market_data
from core.ml_engine import build_feature_frame
from core.ml_sltp import MLStopLossTakeProfit, compute_excursion_targets, pinball_loss
from core.risk_manager import RiskManager
from core.chart_settings import ChartSettings, normalize_timeframe, normalize_chart_ticker, TIMEFRAMES


class TestMLStopLossTakeProfit(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        np.random.seed(7)
        cls.df = generate_synthetic_market_data("BTCUSDT", periods=600)
        cls.model = MLStopLossTakeProfit(cache_file="")  # sin escribir caché en disco
        cls.metrics = cls.model.train(cls.df, "BTCUSDT")
        # Forzar la compuerta de validación para probar la ruta ML de forma determinista
        for side in ("BUY", "SELL"):
            cls.model.metrics_by_key["BTCUSDT"][side]["validated"] = True

    def test_excursion_targets_match_manual_calculation(self):
        feat = build_feature_frame(self.df)
        horizon = 5
        targets = compute_excursion_targets(feat, horizon)
        i = 100
        close = feat['close'].iloc[i]
        atr = max(feat['atr_14'].iloc[i], close * 0.005)
        fut_high = feat['high'].iloc[i + 1:i + 1 + horizon].max()
        fut_low = feat['low'].iloc[i + 1:i + 1 + horizon].min()
        self.assertAlmostEqual(targets['long_mfe'].iloc[i], max(0.0, (fut_high - close) / atr), places=6)
        self.assertAlmostEqual(targets['long_mae'].iloc[i], max(0.0, (close - fut_low) / atr), places=6)
        self.assertAlmostEqual(targets['short_mfe'].iloc[i], targets['long_mae'].iloc[i], places=6)
        # Las últimas `horizon` velas no tienen futuro completo -> NaN (sin fuga de datos)
        self.assertTrue(targets.iloc[-horizon:].isna().all().all())

    def test_pinball_loss(self):
        y = np.array([1.0, 2.0, 3.0])
        self.assertAlmostEqual(pinball_loss(y, y, 0.75), 0.0)
        # Sub-predecir cuesta q, sobre-predecir cuesta (1-q)
        self.assertAlmostEqual(pinball_loss(np.array([2.0]), np.array([1.0]), 0.75), 0.75)
        self.assertAlmostEqual(pinball_loss(np.array([1.0]), np.array([2.0]), 0.75), 0.25)

    def test_training_reports_out_of_sample_metrics(self):
        for side in ("BUY", "SELL"):
            m = self.metrics[side]
            self.assertGreater(m["test_size"], 0)
            self.assertTrue(0.0 <= m["sl_coverage"] <= 1.0)
            self.assertTrue(0.0 <= m["tp_hit_rate"] <= 1.0)
            self.assertIn("sl_pinball_fixed_atr", m)

    def test_predicted_multipliers_respect_bounds_and_min_rr(self):
        for side in ("BUY", "SELL"):
            res = self.model.predict_multipliers(self.df, side, "BTCUSDT")
            self.assertIsNotNone(res)
            self.assertTrue(self.model.sl_bounds[0] <= res["sl_mult"] <= self.model.sl_bounds[1])
            self.assertGreaterEqual(res["tp_mult"], res["sl_mult"] * self.model.min_risk_reward - 1e-6)
        self.assertIsNone(self.model.predict_multipliers(self.df, "HOLD", "BTCUSDT"))

    def test_validation_gate_falls_back_when_ml_not_better(self):
        self.model.metrics_by_key["BTCUSDT"]["SELL"]["validated"] = False
        try:
            self.assertIsNone(self.model.predict_multipliers(self.df, "SELL", "BTCUSDT"))
            rm = RiskManager(sltp_model=self.model)
            price = float(self.df['close'].iloc[-1])
            risk = rm.calculate_trade_parameters(10000.0, price, price * 0.01, "SELL", "BTCUSDT", df=self.df)
            self.assertEqual(risk["sltp_method"], "ATR_FIXED")
            # El otro lado sigue validado
            self.assertIsNotNone(self.model.predict_multipliers(self.df, "BUY", "BTCUSDT"))
        finally:
            self.model.metrics_by_key["BTCUSDT"]["SELL"]["validated"] = True

    def test_validated_flag_reflects_baseline_comparison(self):
        fresh = MLStopLossTakeProfit(cache_file="")
        metrics = fresh.train(self.df, "X")
        for side in ("BUY", "SELL"):
            m = metrics[side]
            expected = m["sl_pinball_model"] < m["sl_pinball_fixed_atr"] and m["tp_pinball_model"] < m["tp_pinball_fixed_atr"]
            # Las métricas se redondean a 5 decimales; sólo comprobar cuando la diferencia es clara
            if abs(m["sl_pinball_model"] - m["sl_pinball_fixed_atr"]) > 1e-4 and abs(m["tp_pinball_model"] - m["tp_pinball_fixed_atr"]) > 1e-4:
                self.assertEqual(m["validated"], expected)

    def test_insufficient_data_falls_back(self):
        tiny = generate_synthetic_market_data("NVDA", periods=60)
        fresh = MLStopLossTakeProfit(cache_file="")
        self.assertIsNone(fresh.predict_multipliers(tiny, "BUY", "NVDA"))

    def test_risk_manager_uses_ml_when_data_available(self):
        rm = RiskManager(sltp_model=self.model)
        price = float(self.df['close'].iloc[-1])
        risk = rm.calculate_trade_parameters(10000.0, price, price * 0.01, "BUY", "BTCUSDT", df=self.df)
        self.assertTrue(risk["valid"])
        self.assertEqual(risk["sltp_method"], "ML")
        self.assertLess(risk["stop_loss"], price)
        self.assertGreater(risk["take_profit"], price)
        self.assertGreaterEqual(risk["risk_reward_ratio"], 1.2 - 0.01)

        sell = rm.calculate_trade_parameters(10000.0, price, price * 0.01, "SELL", "BTCUSDT", df=self.df)
        self.assertGreater(sell["stop_loss"], price)
        self.assertLess(sell["take_profit"], price)

    def test_risk_manager_fixed_atr_fallbacks(self):
        price = 100.0
        # Sin df -> ATR fijo
        rm = RiskManager(sltp_model=self.model)
        risk = rm.calculate_trade_parameters(10000.0, price, 2.0, "BUY", "AAPL")
        self.assertEqual(risk["sltp_method"], "ATR_FIXED")
        self.assertAlmostEqual(risk["stop_loss"], price - 2.0 * 1.5, places=4)
        # Desactivado por el usuario -> ATR fijo aunque haya df
        rm_off = RiskManager(sltp_model=self.model, use_ml_sltp=False)
        risk_off = rm_off.calculate_trade_parameters(10000.0, price, 2.0, "BUY", "BTCUSDT", df=self.df)
        self.assertEqual(risk_off["sltp_method"], "ATR_FIXED")


class TestChartSettings(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmpdir.name, "chart_settings.json")
        self.cs = ChartSettings(path=self.path)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_timeframe_normalization(self):
        self.assertEqual(normalize_timeframe("4H"), "4h")
        self.assertEqual(normalize_timeframe("240"), "4h")
        self.assertEqual(normalize_timeframe("diario"), "1d")
        self.assertEqual(normalize_timeframe("1wk"), "1w")
        self.assertIsNone(normalize_timeframe("3h"))
        self.assertIsNone(normalize_timeframe(""))

    def test_ticker_validation(self):
        self.assertEqual(normalize_chart_ticker(" nvda "), "NVDA")
        self.assertEqual(normalize_chart_ticker("btc-usd"), "BTC-USD")
        self.assertIsNone(normalize_chart_ticker("<script>"))
        self.assertIsNone(normalize_chart_ticker("A" * 25))

    def test_defaults_when_no_file(self):
        data = self.cs.get()
        self.assertEqual(data["timeframe"], "1h")
        self.assertEqual(data["tv_interval"], "60")
        self.assertEqual(data["updated_at"], 0.0)

    def test_update_persists_across_instances(self):
        self.cs.update(ticker="nvda", timeframe="4h", source="telegram")
        other = ChartSettings(path=self.path)  # simula otro proceso (web vs bot)
        data = other.get()
        self.assertEqual(data["ticker"], "NVDA")
        self.assertEqual(data["timeframe"], "4h")
        self.assertEqual(data["tv_interval"], TIMEFRAMES["4h"]["tv"])
        self.assertGreater(data["updated_at"], 0)

        # Actualización parcial conserva el otro campo
        other.update(timeframe="15m")
        self.assertEqual(self.cs.get()["ticker"], "NVDA")
        self.assertEqual(self.cs.get()["timeframe"], "15m")

    def test_invalid_update_raises_and_keeps_state(self):
        self.cs.update(ticker="ETHUSDT", timeframe="1d")
        with self.assertRaises(ValueError):
            self.cs.update(timeframe="7h")
        with self.assertRaises(ValueError):
            self.cs.update(ticker="bad ticker!")
        self.assertEqual(self.cs.get()["timeframe"], "1d")
        self.assertEqual(self.cs.get()["ticker"], "ETHUSDT")

    def test_corrupt_file_falls_back_to_defaults(self):
        with open(self.path, "w") as f:
            f.write("{not json")
        self.assertEqual(self.cs.get()["timeframe"], "1h")


if __name__ == '__main__':
    unittest.main()
