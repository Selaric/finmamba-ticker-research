import unittest

import numpy as np
import pandas as pd

from research_ticker_pairs import (
    evaluate_score_buckets,
    fit_volatility_regimes,
    screen_inverse_pairs,
)


class TickerResearchTests(unittest.TestCase):
    def test_finds_persistent_inverse_pair_using_training_returns(self) -> None:
        random = np.random.default_rng(11)
        first_returns = random.normal(0, 0.01, 240)
        second_returns = -first_returns
        third_returns = random.normal(0, 0.01, 240)
        dates = pd.bdate_range("2018-01-01", periods=240)
        returns = pd.DataFrame(
            {"AAA": first_returns, "BBB": second_returns, "CCC": third_returns},
            index=dates,
        )

        results = screen_inverse_pairs(
            returns,
            train_end="2019-12-31",
            window=30,
            min_windows=4,
        )
        inverse_pair = results.query("ticker_a == 'AAA' and ticker_b == 'BBB'").iloc[0]

        self.assertEqual(inverse_pair["negative_window_fraction"], 1.0)
        self.assertTrue(inverse_pair["persistent_inverse"])
        self.assertAlmostEqual(inverse_pair["latest_train_correlation"], -1.0, places=8)

    def test_inverse_pair_is_limited_to_its_training_regime(self) -> None:
        random = np.random.default_rng(19)
        first_returns = random.normal(0, 0.01, 240)
        second_returns = np.concatenate(
            [-first_returns[:120], first_returns[120:]]
        )
        dates = pd.bdate_range("2018-01-01", periods=240)
        returns = pd.DataFrame(
            {"AAA": first_returns, "BBB": second_returns}, index=dates
        )
        regimes = pd.Series(["low"] * 120 + ["high"] * 120, index=dates)

        results = screen_inverse_pairs(
            returns,
            train_end="2019-12-31",
            regimes=regimes,
            window=30,
            min_windows=4,
        ).set_index("discovered_regime")

        self.assertTrue(results.loc["low", "persistent_inverse"])
        self.assertFalse(results.loc["high", "persistent_inverse"])

    def test_regime_cutoffs_use_training_period_only(self) -> None:
        dates = pd.bdate_range("2020-01-01", periods=100)
        historical = pd.Series(np.sin(np.arange(100) / 4) * 0.01, index=dates)
        baseline_regimes, baseline_cutoffs = fit_volatility_regimes(
            historical, train_end="2020-03-25", window=5
        )
        changed_test = historical.copy()
        changed_test.loc[changed_test.index > pd.Timestamp("2020-03-25")] *= 100
        changed_regimes, changed_cutoffs = fit_volatility_regimes(
            changed_test, train_end="2020-03-25", window=5
        )

        self.assertEqual(baseline_cutoffs, changed_cutoffs)
        self.assertTrue(baseline_regimes.loc[:"2020-03-25"].equals(
            changed_regimes.loc[:"2020-03-25"]
        ))

    def test_rank_ic_and_buckets_reflect_perfect_cross_sectional_order(self) -> None:
        dates = pd.bdate_range("2024-01-01", periods=3)
        tickers = ["AAA", "BBB", "CCC", "DDD", "EEE"]
        scores = pd.DataFrame([range(5)] * 3, index=dates, columns=tickers)
        forward_returns = scores / 100
        regimes = pd.Series(["low", "mid", "high"], index=dates)

        daily_ic, ic_summary, buckets = evaluate_score_buckets(
            scores,
            forward_returns,
            regimes,
            start_date="2024-01-02",
            bucket_count=5,
        )

        self.assertTrue(np.allclose(daily_ic["rank_ic"], 1.0))
        self.assertEqual(set(ic_summary["regime"]), {"mid", "high"})
        low_to_high = buckets.sort_values("score_bucket")["mean_forward_return"].to_list()
        self.assertEqual(low_to_high, sorted(low_to_high))


if __name__ == "__main__":
    unittest.main()