from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from research_ticker_pairs import (
    evaluate_pair_regimes,
    evaluate_score_buckets,
    fit_volatility_regimes,
    screen_inverse_pairs,
)


def evaluate_outputs(
    *,
    data_dir: Path,
    prices_path: Path,
    output_dir: Path,
    train_start: str,
    train_end: str,
    test_start: str,
    volatility_window: int = 20,
    relation_window: int = 60,
    momentum_window: int = 20,
    bucket_count: int = 5,
) -> dict[str, pd.DataFrame | dict[str, float]]:
    label_path = data_dir / "nasdaqlab.pkl"
    if not label_path.exists():
        raise FileNotFoundError(f"FinMamba target labels not found: {label_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_labels = pd.read_pickle(label_path).reset_index()
    raw_labels["datetime"] = pd.to_datetime(raw_labels["datetime"])
    labels = raw_labels.pivot(index="datetime", columns="instrument", values="label").sort_index()
    tickers = list(labels.columns)

    scores_array = pd.read_csv(output_dir / "scores.csv", header=None).to_numpy()
    prediction_rows = raw_labels.loc[
        raw_labels["datetime"] >= pd.Timestamp(test_start)
    ]
    test_dates = pd.DatetimeIndex(sorted(prediction_rows["datetime"].unique()))
    if scores_array.shape != (len(test_dates), len(tickers)):
        raise ValueError(
            f"Scores have shape {scores_array.shape}; expected "
            f"({len(test_dates)} test dates, {len(tickers)} tickers)."
        )

    targets = labels.reindex(index=test_dates, columns=tickers)
    model_scores = pd.DataFrame(scores_array, index=test_dates, columns=tickers)
    prices = pd.read_csv(prices_path, index_col=0, parse_dates=True).sort_index()
    missing_prices = sorted(set(tickers + ["SPY"]) - set(prices.columns))
    if missing_prices:
        raise ValueError(f"Adjusted close file is missing tickers: {missing_prices}")

    benchmark_returns = prices["SPY"].pct_change(fill_method=None)
    benchmark_returns = benchmark_returns.loc[
        benchmark_returns.index >= pd.Timestamp(train_start)
    ]
    regimes, cutoffs = fit_volatility_regimes(
        benchmark_returns,
        train_end=train_end,
        window=volatility_window,
    )
    momentum_scores = prices[tickers].pct_change(
        periods=momentum_window,
        fill_method=None,
    ).reindex(test_dates)
    score_frames = {
        "finmamba_reference": model_scores,
        "momentum_20d": momentum_scores,
        "mean_reversion_20d": -momentum_scores,
    }

    daily_ic_frames: list[pd.DataFrame] = []
    ic_summary_frames: list[pd.DataFrame] = []
    bucket_frames: list[pd.DataFrame] = []
    for model_name, score_frame in score_frames.items():
        daily_ic, ic_summary, buckets = evaluate_score_buckets(
            score_frame,
            targets,
            regimes,
            start_date=test_start,
            bucket_count=bucket_count,
        )
        daily_ic["model"] = model_name
        ic_summary["model"] = model_name
        buckets["model"] = model_name
        daily_ic_frames.append(daily_ic)
        ic_summary_frames.append(ic_summary)
        bucket_frames.append(buckets)

    daily_ic_results = pd.concat(daily_ic_frames, ignore_index=True)
    ic_results = pd.concat(ic_summary_frames, ignore_index=True)
    bucket_results = pd.concat(bucket_frames, ignore_index=True)

    returns = prices[tickers].pct_change(fill_method=None)
    returns = returns.loc[returns.index >= pd.Timestamp(train_start)]
    pair_screen = screen_inverse_pairs(
        returns,
        train_end=train_end,
        regimes=regimes,
        window=relation_window,
    )
    test_returns = returns.loc[returns.index >= pd.Timestamp(test_start)]
    test_regimes = regimes.loc[regimes.index >= pd.Timestamp(test_start)]
    pair_holdout = evaluate_pair_regimes(
        pair_screen,
        test_returns,
        test_regimes,
    )
    pair_summary = (
        pair_holdout.groupby(["discovered_regime", "test_regime"], as_index=False)
        .agg(
            pairs=("test_correlation", "count"),
            train_candidates=("train_persistent_inverse", "sum"),
            mean_test_correlation=("test_correlation", "mean"),
        )
    )

    artifacts = {
        "finmamba_daily_ic.csv": daily_ic_results,
        "finmamba_ic_summary.csv": ic_results,
        "finmamba_score_buckets.csv": bucket_results,
        "finmamba_pair_screen.csv": pair_screen,
        "finmamba_pair_holdout.csv": pair_holdout,
        "finmamba_pair_regime_summary.csv": pair_summary,
    }
    for filename, frame in artifacts.items():
        frame.to_csv(output_dir / filename, index=False)

    return {
        "ic_summary": ic_results,
        "bucket_summary": bucket_results,
        "pair_screen": pair_screen,
        "pair_regime_summary": pair_summary,
        "regime_cutoffs": cutoffs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate FinMamba holdout predictions by SPY volatility regime."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--prices", type=Path, default=Path("outputs/ticker_research_nasdaq100/adjusted_closes.csv")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/finmamba_reference"))
    parser.add_argument("--train-start", default="2018-01-01")
    parser.add_argument("--train-end", default="2021-12-31")
    parser.add_argument("--test-start", default="2023-01-01")
    parser.add_argument("--volatility-window", type=int, default=20)
    parser.add_argument("--relation-window", type=int, default=60)
    parser.add_argument("--momentum-window", type=int, default=20)
    parser.add_argument("--buckets", type=int, default=5)
    args = parser.parse_args()

    results = evaluate_outputs(
        data_dir=args.data_dir,
        prices_path=args.prices,
        output_dir=args.output_dir,
        train_start=args.train_start,
        train_end=args.train_end,
        test_start=args.test_start,
        volatility_window=args.volatility_window,
        relation_window=args.relation_window,
        momentum_window=args.momentum_window,
        bucket_count=args.buckets,
    )
    print(f"Train-fitted SPY volatility cutoffs: {results['regime_cutoffs']}")
    print("Daily rank IC summary:")
    print(results["ic_summary"].to_string(index=False))
    print("Mean next-five-day excess return by score bucket:")
    print(results["bucket_summary"].to_string(index=False))
    persistent_pairs = int(results["pair_screen"]["persistent_inverse"].sum())
    print(f"Persistent inverse training pair/regime rows: {persistent_pairs}")
    print("Holdout pair correlations:")
    print(results["pair_regime_summary"].to_string(index=False))


if __name__ == "__main__":
    main()