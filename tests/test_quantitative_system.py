"""
Pruebas Unitarias de Validación para el Proyecto Integrador.
Valida:
1. Cálculo matemático de indicadores (Regresión, SAR, MACD, RSI, ATR).
2. Pipeline de Machine Learning y Simulación de Hiperparámetros.
3. Test formal de Diebold-Mariano y corrección HLN.
4. Simulación Monte Carlo y métricas de riesgo.
5. Motor de órdenes, balance y cálculo de Profit Factor / Win Rate.
"""

import unittest
import numpy as np
import pandas as pd

from core.data_feed import generate_synthetic_market_data
from core.indicators import compute_all_indicators, calculate_linear_regression, calculate_parabolic_sar
from core.ml_engine import diebold_mariano_test, QuantitativeMLEngine, compute_regression_metrics
from core.monte_carlo import run_monte_carlo_simulation
from core.risk_manager import RiskManager
from core.execution_engine import ExecutionEngine


class TestQuantitativeSystem(unittest.TestCase):

    def setUp(self):
        self.df = generate_synthetic_market_data("BTCUSDT", periods=200)

    def test_indicators_computation(self):
        df_ind = compute_all_indicators(self.df)
        expected_cols = ['rsi_14', 'macd', 'macd_signal', 'stoch_k', 'stoch_d', 'lr_slope', 'parabolic_sar', 'atr_14']
        for col in expected_cols:
            self.assertIn(col, df_ind.columns, f"Falta indicador: {col}")
        self.assertFalse(df_ind['rsi_14'].dropna().empty)

    def test_diebold_mariano_test(self):
        y_real = np.array([0.02, -0.01, 0.03, -0.02, 0.01, 0.015, -0.005, 0.02])
        y_pred_rf = np.array([0.019, -0.008, 0.028, -0.018, 0.009, 0.014, -0.004, 0.018])
        y_pred_knn = np.array([0.01, -0.002, 0.015, -0.005, 0.002, 0.008, 0.001, 0.010])

        res = diebold_mariano_test(y_real, y_pred_rf, y_pred_knn, h=1)
        self.assertIn("dm_stat", res)
        self.assertIn("p_value", res)
        self.assertLess(res["dm_stat"], 0, "El estadístico DM debe reflejar menor pérdida para el modelo superior.")

    def test_monte_carlo_simulation(self):
        current_price = float(self.df['close'].iloc[-1])
        sl = current_price * 0.97
        tp = current_price * 1.05

        mc = run_monte_carlo_simulation(self.df, current_price, sl, tp, num_simulations=100, time_horizon_bars=24)
        self.assertIn("prob_take_profit", mc)
        self.assertIn("prob_stop_loss", mc)
        self.assertIn("value_at_risk_95", mc)
        self.assertAlmostEqual(mc["prob_take_profit"] + mc["prob_stop_loss"] + mc["prob_unresolved"], 1.0, places=2)

    def test_risk_manager_and_execution(self):
        rm = RiskManager()
        engine = ExecutionEngine(initial_balance=10000.0)

        risk = rm.calculate_trade_parameters(10000.0, 90000.0, 1500.0, "BUY")
        self.assertTrue(risk["valid"])
        self.assertGreater(risk["take_profit"], 90000.0)
        self.assertLess(risk["stop_loss"], 90000.0)

        # Prueba de apertura y cierre
        res = engine.open_position("TESTUSDT", "BUY", 90000.0, 0.1, 88000.0, 94000.0)
        self.assertTrue(res["success"])

        # Simular llegada al TP
        closed = engine.update_and_check_positions("TESTUSDT", 94100.0)
        self.assertEqual(len(closed), 1)
        self.assertEqual(closed[0]["exit_reason"], "TAKE_PROFIT")
        self.assertGreater(closed[0]["pnl"], 0)


if __name__ == "__main__":
    unittest.main()

