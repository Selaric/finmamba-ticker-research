import unittest

import numpy as np
import pandas as pd

from prepare_finmamba_data import build_panels


class PrepareFinMambaDataTests(unittest.TestCase):
    def test_builds_balanced_train_scaled_panels_and_relative_labels(self) -> None:
        random = np.random.default_rng(23)
        dates = pd.bdate_range("2020-01-01", periods=160)
        tickers = [f"T{index:02d}" for index in range(20)]
        close = pd.DataFrame(
            100 * np.exp(np.cumsum(random.normal(0, 0.01, (len(dates), len(tickers))), axis=0)),
            index=dates,
            columns=tickers,
        )
        raw_fields = {
            "Open": close * (1 + random.normal(0, 0.002, close.shape)),
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": pd.DataFrame(
                random.integers(1000, 10000, close.shape),
                index=dates,
                columns=tickers,
            ),
        }
        raw = pd.concat(raw_fields, axis=1, names=["Price", "Ticker"])

        features, labels, included, feature_names = build_panels(
            raw,
            tickers,
            train_end="2020-06-30",
            horizon=5,
        )

        self.assertEqual(included, tickers)
        self.assertEqual(len(feature_names), 5)
        self.assertEqual(len(features), len(labels))
        self.assertFalse(features[feature_names].isna().any().any())
        self.assertFalse(labels["label"].isna().any())

        training_mask = features["datetime"] <= pd.Timestamp("2020-06-30")
        training_means = features.loc[training_mask].groupby("instrument")[
            feature_names
        ].mean()
        self.assertTrue(np.allclose(training_means.to_numpy(), 0, atol=1e-10))
        label_means = labels.groupby("datetime")["label"].mean()
        self.assertTrue(np.allclose(label_means.to_numpy(), 0, atol=1e-12))


if __name__ == "__main__":
    unittest.main()