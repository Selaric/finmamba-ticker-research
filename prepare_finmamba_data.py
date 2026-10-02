from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from research_ticker_pairs import fetch_nasdaq100_tickers


FEATURE_FIELDS = ("Open", "High", "Low", "Close", "Volume")


def _extract_field(raw: pd.DataFrame, field: str, tickers: list[str]) -> pd.DataFrame:
    values = raw[field]
    if isinstance(values, pd.Series):
        values = values.to_frame(name=tickers[0])
    missing = sorted(set(tickers) - set(values.columns))
    if missing:
        raise ValueError(f"Adjusted OHLCV data is missing {field} for: {missing}")
    return values.loc[:, tickers].astype(float)


def build_panels(
    raw: pd.DataFrame,
    tickers: list[str],
    *,
    train_end: str,
    horizon: int,
    minimum_stocks: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    """Build complete date-major panels with train-only per-ticker scaling."""
    if horizon < 1:
        raise ValueError("horizon must be at least one trading day")

    fields = {field: _extract_field(raw, field, tickers) for field in FEATURE_FIELDS}
    close = fields["Close"]
    complete_tickers = [
        ticker
        for ticker in tickers
        if all(fields[field][ticker].notna().all() for field in FEATURE_FIELDS)
    ]
    if len(complete_tickers) < minimum_stocks:
        raise ValueError(
            f"Only {len(complete_tickers)} tickers have complete history; "
            f"need at least {minimum_stocks}. Use a later --start date or smaller universe."
        )

    fields = {field: values[complete_tickers] for field, values in fields.items()}
    close = fields["Close"]
    previous_close = close.shift(1)
    feature_wide = pd.concat(
        {
            "open_return": fields["Open"].div(previous_close).sub(1),
            "high_return": fields["High"].div(previous_close).sub(1),
            "low_return": fields["Low"].div(previous_close).sub(1),
            "close_return": close.pct_change(fill_method=None),
            "log_volume": np.log1p(fields["Volume"].clip(lower=0)),
        },
        axis=1,
        names=["feature", "ticker"],
    )

    forward_returns = close.shift(-horizon).div(close).sub(1)
    labels_wide = forward_returns.sub(forward_returns.mean(axis=1), axis=0)
    complete_dates = feature_wide.notna().all(axis=1) & labels_wide.notna().all(axis=1)
    feature_wide = feature_wide.loc[complete_dates]
    labels_wide = labels_wide.loc[complete_dates]

    training_rows = feature_wide.index <= pd.Timestamp(train_end)
    if training_rows.sum() < 100:
        raise ValueError("Fewer than 100 complete training dates for feature scaling")
    means = feature_wide.loc[training_rows].mean(axis=0)
    standard_deviations = feature_wide.loc[training_rows].std(axis=0).replace(0, 1)
    feature_wide = (feature_wide - means) / standard_deviations

    feature_names = list(feature_wide.columns.get_level_values("feature").unique())
    feature_records: list[pd.DataFrame] = []
    label_records: list[pd.DataFrame] = []
    for ticker in complete_tickers:
        ticker_features = feature_wide.xs(ticker, axis=1, level="ticker").copy()
        ticker_features["datetime"] = feature_wide.index
        ticker_features["instrument"] = ticker
        feature_records.append(ticker_features)

        ticker_labels = labels_wide[[ticker]].rename(columns={ticker: "label"})
        ticker_labels["datetime"] = labels_wide.index
        ticker_labels["instrument"] = ticker
        label_records.append(ticker_labels)

    feature_frame = pd.concat(feature_records, ignore_index=True).sort_values(
        ["datetime", "instrument"], ignore_index=True
    )
    label_frame = pd.concat(label_records, ignore_index=True).sort_values(
        ["datetime", "instrument"], ignore_index=True
    )
    if feature_frame[feature_names].isna().any().any() or label_frame["label"].isna().any():
        raise ValueError("Prepared feature and label panels must not contain missing values")
    return feature_frame, label_frame, complete_tickers, feature_names


def download_adjusted_ohlcv(
    tickers: list[str], *, start_date: str, end_date: str
) -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if raw.empty or not isinstance(raw.columns, pd.MultiIndex):
        raise RuntimeError("Yahoo Finance returned no multi-ticker OHLCV data")
    return raw


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare balanced FinMamba data panels from adjusted OHLCV prices."
    )
    parser.add_argument("--tickers", nargs="+", default=None)
    parser.add_argument("--benchmark", default="SPY")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2026-10-02", help="Exclusive end date")
    parser.add_argument("--train-end", default="2021-12-31")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--minimum-stocks", type=int, default=20)
    parser.add_argument("--stock-name", default="nasdaq")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    if args.tickers is None:
        universe, universe_as_of = fetch_nasdaq100_tickers()
    else:
        universe = list(dict.fromkeys(args.tickers))
        universe_as_of = "user-specified"
    universe = [ticker for ticker in universe if ticker != args.benchmark]
    download_tickers = [*universe, args.benchmark]
    raw = download_adjusted_ohlcv(
        download_tickers,
        start_date=args.start,
        end_date=args.end,
    )
    feature_frame, label_frame, complete_tickers, feature_names = build_panels(
        raw,
        universe,
        train_end=args.train_end,
        horizon=args.horizon,
        minimum_stocks=args.minimum_stocks,
    )

    args.data_dir.mkdir(parents=True, exist_ok=True)
    feature_frame.set_index(["datetime", "instrument"]).to_pickle(
        args.data_dir / f"{args.stock_name}fea.pkl"
    )
    label_frame.set_index(["datetime", "instrument"]).to_pickle(
        args.data_dir / f"{args.stock_name}lab.pkl"
    )
    np.save(
        args.data_dir / f"{args.stock_name}_industry_relationship.npy",
        np.ones((len(complete_tickers), len(complete_tickers)), dtype=np.float32),
    )
    pd.DataFrame(
        {
            "ticker": complete_tickers,
            "universe_as_of": universe_as_of,
            "benchmark": args.benchmark,
            "label_horizon_days": args.horizon,
        }
    ).to_csv(args.data_dir / f"{args.stock_name}_model_universe.csv", index=False)
    print(f"Complete-history tickers: {len(complete_tickers)} of {len(universe)}")
    print(f"Dates: {feature_frame['datetime'].min()} to {feature_frame['datetime'].max()}")
    print(f"Features: {feature_names}")
    print(f"Target: next-{args.horizon}-day return minus same-day universe mean")
    print(f"Saved FinMamba panels under {args.data_dir}")


if __name__ == "__main__":
    main()