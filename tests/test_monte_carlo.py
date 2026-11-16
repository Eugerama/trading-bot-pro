"""
Pruebas de la simulación Monte Carlo (core/monte_carlo.py) y su uso en la estrategia.
"""

import math
import unittest

import numpy as np
import pandas as pd

from core.monte_carlo import run_monte_carlo_simulation, MAX_DRIFT_SIGMA_FRACTION
from core.data_feed import generate_synthetic_market_data
from core.strategy import HybridQuantStrategy


def make_df(log_returns: np.ndarray, start: float = 100.0) -> pd.DataFrame:
    close = start * np.exp(np.concatenate([[0.0], np.cumsum(log_returns)]))
    idx = pd.date_range("2026-01-01", periods=len(close), freq="h")
    return pd.DataFrame({
        "open": close, "high": close * 1.001, "low": close * 0.999, "close": close, "volume": 1000.0
    }, index=idx)


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


class TestMonteCarlo(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(1)
        # Paseo aleatorio sin tendencia, sigma ~ 1% por vela
        self.flat_df = make_df(rng.normal(0.0, 0.01, 2000))
        # Tendencia fuerte: +0.5% por vela (mucho mayor que el tope de drift)
        self.trend_df = make_df(rng.normal(0.005, 0.01, 2000))

    def test_does_not_touch_global_rng(self):
        np.random.seed(123)
        expected = np.random.rand()
        np.random.seed(123)
        run_monte_carlo_simulation(self.flat_df, 100.0, 97.0, 105.0)
        self.assertEqual(np.random.rand(), expected)

    def test_reproducible_with_same_seed(self):
        a = run_monte_carlo_simulation(self.flat_df, 100.0, 97.0, 105.0, seed=7)
        b = run_monte_carlo_simulation(self.flat_df, 100.0, 97.0, 105.0, seed=7)
        self.assertEqual(a["prob_take_profit"], b["prob_take_profit"])
        self.assertEqual(a["expected_return"], b["expected_return"])

    def test_drift_is_shrunk_and_capped(self):
        res = run_monte_carlo_simulation(self.trend_df, 100.0, 97.0, 105.0)
        sigma = res["sigma_per_bar"]
        self.assertGreater(res["historical_drift_per_bar"], 0.004)
        self.assertAlmostEqual(res["drift_per_bar_used"], MAX_DRIFT_SIGMA_FRACTION * sigma, places=6)

        zero = run_monte_carlo_simulation(self.trend_df, 100.0, 97.0, 105.0, drift_shrinkage=0.0)
        self.assertEqual(zero["drift_per_bar_used"], 0.0)

        small_trend = make_df(np.random.default_rng(3).normal(0.0002, 0.01, 2000))
        res_small = run_monte_carlo_simulation(small_trend, 100.0, 97.0, 105.0, drift_shrinkage=0.25)
        self.assertAlmostEqual(res_small["drift_per_bar_used"], 0.25 * res_small["historical_drift_per_bar"], places=7)

    def test_no_double_ito_correction(self):
        # Con drift 0 en log-precio, la mediana del precio final debe quedar ≈ en la entrada
        res = run_monte_carlo_simulation(self.flat_df, 100.0, 1.0, 10_000.0, drift_shrinkage=0.0,
                                         num_simulations=4000, time_horizon_bars=48)
        median_final = res["percentiles"]["p50"][-1]
        self.assertAlmostEqual(median_final, 100.0, delta=1.0)

    def test_stop_hit_probability_matches_reflection_principle(self):
        # Barrera única (TP inalcanzable): P(tocar SL en H velas) = 2 * Φ(ln(SL/S0) / (σ√H)) para BM sin drift
        H = 36
        res = run_monte_carlo_simulation(self.flat_df, 100.0, 97.0, 10_000.0, drift_shrinkage=0.0,
                                         num_simulations=6000, time_horizon_bars=H, seed=11)
        sigma = res["sigma_per_bar"]
        theoretical = 2.0 * normal_cdf(math.log(97.0 / 100.0) / (sigma * math.sqrt(H)))
        self.assertAlmostEqual(res["prob_stop_loss"], theoretical, delta=0.025)

    def test_probabilities_sum_to_one_for_both_sides(self):
        for side, sl, tp in (("BUY", 97.0, 104.0), ("SELL", 103.0, 96.0)):
            res = run_monte_carlo_simulation(self.flat_df, 100.0, sl, tp, position_type=side, num_simulations=500)
            total = res["prob_take_profit"] + res["prob_stop_loss"] + res["prob_unresolved"]
            self.assertAlmostEqual(total, 1.0, places=3)
            self.assertLessEqual(res["value_at_risk_95"], res["expected_return"])

    def test_sell_mirrors_buy_without_drift(self):
        buy = run_monte_carlo_simulation(self.flat_df, 100.0, 97.0, 103.0, "BUY", 4000, 36, drift_shrinkage=0.0)
        sell = run_monte_carlo_simulation(self.flat_df, 100.0, 103.0, 97.0, "SELL", 4000, 36, drift_shrinkage=0.0)
        # En log-precio 97 y 103 no son exactamente simétricos; basta con estar cerca
        self.assertAlmostEqual(buy["prob_take_profit"], sell["prob_stop_loss"], delta=0.05)


class _StubRiskManager:
    def __init__(self):
        self.calls = []

    def calculate_trade_parameters(self, balance, price, atr, side, ticker="", df=None):
        self.calls.append((side, ticker, df is not None))
        if side == "BUY":
            return {"valid": True, "stop_loss": price * 0.9, "take_profit": price * 1.2}
        return {"valid": True, "stop_loss": price * 1.1, "take_profit": price * 0.8}


class TestStrategyUsesRealLevels(unittest.TestCase):

    def test_monte_carlo_uses_risk_manager_levels(self):
        df = generate_synthetic_market_data("BTCUSDT", periods=300)
        stub = _StubRiskManager()
        strat = HybridQuantStrategy(risk_manager=stub)
        strat.evaluate_signals(df, {"signal": "HOLD", "confidence": 0.5}, ticker="BTCUSDT")
        self.assertEqual(len(stub.calls), 1)
        side, ticker, has_df = stub.calls[0]
        self.assertIn(side, ("BUY", "SELL"))
        self.assertEqual(ticker, "BTCUSDT")
        self.assertTrue(has_df)


if __name__ == '__main__':
    unittest.main()
