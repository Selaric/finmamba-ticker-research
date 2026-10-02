from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import requests


def fetch_nasdaq100_tickers() -> tuple[list[str], str]:
    response = requests.get(
        "https://api.nasdaq.com/api/quote/list-type/nasdaq100",
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json, text/plain, */*",
            "Origin": "https://www.nasdaq.com",
        },
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()["data"]
    rows = payload["data"]["rows"]
    tickers = sorted({row["symbol"].strip().replace(".", "-") for row in rows})
    if len(tickers) < 90:
        raise RuntimeError(f"Nasdaq API returned only {len(tickers)} Nasdaq-100 symbols")
    return tickers, str(payload.get("date", "unknown"))


def screen_inverse_pairs(
    returns: pd.DataFrame,
    *,
    train_end: str,
    regimes: pd.Series | None = None,
    window: int = 60,
    correlation_threshold: float = -0.3,
    min_negative_fraction: float = 0.75,
    min_windows: int = 4,
) -> pd.DataFrame:
    """Find pairs with consistently negative correlations in training data."""
    if window < 2 or min_windows < 1:
        raise ValueError("window must be >= 2 and min_windows must be >= 1")
    if not -1 <= correlation_threshold < 0:
        raise ValueError("correlation_threshold must be in [-1, 0)")
    if not 0 <= min_negative_fraction <= 1:
        raise ValueError("min_negative_fraction must be between 0 and 1")
    if returns.shape[1] < 2:
        raise ValueError("At least two tickers are required")

    training_returns = returns.loc[pd.to_datetime(returns.index) <= pd.Timestamp(train_end)]
    training_regimes = (
        pd.Series("all", index=training_returns.index, name="regime")
        if regimes is None
        else regimes.reindex(training_returns.index).rename("regime")
    )
    rows: list[dict[str, object]] = []
    ticker_names = list(training_returns.columns)
    min_observations = max(2, int(np.ceil(window * 0.8)))
    for regime_name in training_regimes.dropna().unique():
        regime_dates = training_regimes.index[training_regimes.eq(regime_name)]
        regime_returns = training_returns.loc[regime_dates]
        first_window = len(regime_returns) % window
        aligned_returns = regime_returns.iloc[first_window:]
        correlation_matrices = [
            aligned_returns.iloc[start : start + window].corr(
                min_periods=min_observations
            ).to_numpy()
            for start in range(0, len(aligned_returns), window)
            if len(aligned_returns.iloc[start : start + window]) == window
        ]
        if not correlation_matrices:
            continue
        window_correlations = np.stack(correlation_matrices)
        for ticker_a_index, ticker_b_index in combinations(range(len(ticker_names)), 2):
            ticker_a = ticker_names[ticker_a_index]
            ticker_b = ticker_names[ticker_b_index]
            pair_correlations = window_correlations[:, ticker_a_index, ticker_b_index]
            pair_correlations = pair_correlations[np.isfinite(pair_correlations)]
            if len(pair_correlations) < min_windows:
                continue

            negative_fraction = sum(
                value <= correlation_threshold for value in pair_correlations
            ) / len(pair_correlations)
            rows.append(
                {
                    "ticker_a": ticker_a,
                    "ticker_b": ticker_b,
                    "discovered_regime": regime_name,
                    "latest_train_correlation": pair_correlations[-1],
                    "mean_train_correlation": float(np.mean(pair_correlations)),
                    "negative_window_fraction": negative_fraction,
                    "windows_observed": len(pair_correlations),
                    "persistent_inverse": (
                        pair_correlations[-1] <= correlation_threshold
                        and negative_fraction >= min_negative_fraction
                    ),
                }
            )

    columns = [
        "ticker_a",
        "ticker_b",
        "discovered_regime",
        "latest_train_correlation",
        "mean_train_correlation",
        "negative_window_fraction",
        "windows_observed",
        "persistent_inverse",
    ]
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["persistent_inverse", "mean_train_correlation"],
        ascending=[False, True],
        ignore_index=True,
    )


def fit_volatility_regimes(
    benchmark_returns: pd.Series,
    *,
    train_end: str,
    window: int = 20,
) -> tuple[pd.Series, dict[str, float]]:
    """Label low/mid/high benchmark-volatility regimes using train-only terciles."""
    if window < 2:
        raise ValueError("window must be at least 2")

    annualized_volatility = benchmark_returns.rolling(
        window=window, min_periods=window
    ).std() * np.sqrt(252)
    training_volatility = annualized_volatility.loc[
        annualized_volatility.index <= pd.Timestamp(train_end)
    ].dropna()
    if training_volatility.empty:
        raise ValueError("Not enough training observations to fit volatility regimes")

    low_cutoff, high_cutoff = training_volatility.quantile([1 / 3, 2 / 3])
    regimes = pd.Series(index=annualized_volatility.index, dtype="object")
    valid = annualized_volatility.notna()
    regimes.loc[valid & annualized_volatility.le(low_cutoff)] = "low"
    regimes.loc[
        valid & annualized_volatility.gt(low_cutoff) & annualized_volatility.le(high_cutoff)
    ] = "mid"
    regimes.loc[valid & annualized_volatility.gt(high_cutoff)] = "high"
    return regimes, {"low_cutoff": float(low_cutoff), "high_cutoff": float(high_cutoff)}


def evaluate_pair_regimes(
    candidates: pd.DataFrame,
    test_returns: pd.DataFrame,
    regimes: pd.Series,
    *,
    min_observations: int = 30,
) -> pd.DataFrame:
    """Measure selected pairs' holdout correlations within each regime."""
    columns = [
        "ticker_a",
        "ticker_b",
        "discovered_regime",
        "test_regime",
        "train_persistent_inverse",
        "train_mean_correlation",
        "train_negative_window_fraction",
        "observations",
        "test_correlation",
    ]
    rows: list[dict[str, object]] = []
    for pair in candidates.itertuples(index=False):
        for regime_name in ("low", "mid", "high"):
            dates = regimes.index[regimes.eq(regime_name)]
            pair_data = test_returns.reindex(dates)[[pair.ticker_a, pair.ticker_b]].dropna()
            correlation = (
                float(pair_data.corr().iloc[0, 1])
                if len(pair_data) >= min_observations
                else np.nan
            )
            rows.append(
                {
                    "ticker_a": pair.ticker_a,
                    "ticker_b": pair.ticker_b,
                    "discovered_regime": pair.discovered_regime,
                    "test_regime": regime_name,
                    "train_persistent_inverse": pair.persistent_inverse,
                    "train_mean_correlation": pair.mean_train_correlation,
                    "train_negative_window_fraction": pair.negative_window_fraction,
                    "observations": len(pair_data),
                    "test_correlation": correlation,
                }
            )
    return pd.DataFrame(rows, columns=columns)


def evaluate_score_buckets(
    scores: pd.DataFrame,
    forward_returns: pd.DataFrame,
    regimes: pd.Series,
    *,
    start_date: str,
    bucket_count: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Calculate per-date rank IC and forward-return buckets on a holdout."""
    if bucket_count < 2:
        raise ValueError("bucket_count must be at least 2")

    test_start = pd.Timestamp(start_date)
    dates = scores.index.intersection(forward_returns.index).intersection(regimes.index)
    dates = dates[dates >= test_start]
    ic_rows: list[dict[str, object]] = []
    bucket_rows: list[dict[str, object]] = []

    for date in dates:
        day_scores, day_returns = scores.loc[date].align(
            forward_returns.loc[date], join="inner"
        )
        valid = day_scores.notna() & day_returns.notna()
        day_scores, day_returns = day_scores.loc[valid], day_returns.loc[valid]
        if len(day_scores) < 3:
            continue

        rank_ic = day_scores.corr(day_returns, method="spearman")
        regime_name = regimes.loc[date]
        if pd.notna(rank_ic):
            ic_rows.append(
                {
                    "date": date,
                    "regime": regime_name,
                    "rank_ic": float(rank_ic),
                    "ticker_count": len(day_scores),
                }
            )

        day_bucket_count = min(bucket_count, len(day_scores))
        bucket_ids = pd.qcut(
            day_scores.rank(method="first"),
            q=day_bucket_count,
            labels=False,
        ) + 1
        for bucket_id, ticker_returns in day_returns.groupby(bucket_ids):
            bucket_rows.append(
                {
                    "date": date,
                    "regime": regime_name,
                    "score_bucket": int(bucket_id),
                    "mean_forward_return": float(ticker_returns.mean()),
                    "ticker_count": len(ticker_returns),
                }
            )

    daily_ic = pd.DataFrame(ic_rows, columns=["date", "regime", "rank_ic", "ticker_count"])
    if daily_ic.empty:
        ic_summary = pd.DataFrame(
            columns=["regime", "days", "mean_rank_ic", "median_rank_ic"]
        )
    else:
        ic_summary = (
            daily_ic.groupby("regime", as_index=False)
            .agg(
                days=("rank_ic", "size"),
                mean_rank_ic=("rank_ic", "mean"),
                median_rank_ic=("rank_ic", "median"),
            )
            .sort_values("regime", ignore_index=True)
        )

    daily_buckets = pd.DataFrame(
        bucket_rows,
        columns=["date", "regime", "score_bucket", "mean_forward_return", "ticker_count"],
    )
    if daily_buckets.empty:
        bucket_summary = pd.DataFrame(
            columns=["regime", "score_bucket", "days", "mean_forward_return"]
        )
    else:
        bucket_summary = (
            daily_buckets.groupby(["regime", "score_bucket"], as_index=False)
            .agg(
                days=("date", "nunique"),
                mean_forward_return=("mean_forward_return", "mean"),
            )
            .sort_values(["regime", "score_bucket"], ignore_index=True)
        )
    return daily_ic, ic_summary, bucket_summary


def analyze_prices(
    prices: pd.DataFrame,
    *,
    benchmark: str = "SPY",
    train_end: str = "2019-12-31",
    test_start: str = "2020-01-01",
    relation_window: int = 60,
    regime_window: int = 20,
    score_lookback: int = 20,
    bucket_count: int = 5,
) -> dict[str, pd.DataFrame | dict[str, float]]:
    """Run train-only pair discovery and holdout regime/IC diagnostics."""
    if benchmark not in prices.columns:
        raise ValueError(f"Benchmark ticker {benchmark!r} is missing from prices")
    tickers = [ticker for ticker in prices.columns if ticker != benchmark]
    if len(tickers) < 3:
        raise ValueError("At least three non-benchmark tickers are required")

    prices = prices.sort_index().apply(pd.to_numeric, errors="raise")
    returns = prices.pct_change(fill_method=None)
    training_end = pd.Timestamp(train_end)
    test_start_date = pd.Timestamp(test_start)
    regimes, regime_cutoffs = fit_volatility_regimes(
        returns[benchmark], train_end=train_end, window=regime_window
    )
    pair_screen = screen_inverse_pairs(
        returns[tickers],
        train_end=train_end,
        regimes=regimes,
        window=relation_window,
    )

    test_mask = returns.index >= test_start_date
    test_returns = returns.loc[test_mask, tickers]
    test_regimes = regimes.loc[test_mask]
    pair_regime_results = evaluate_pair_regimes(
        pair_screen, test_returns, test_regimes
    )

    forward_returns = prices[tickers].shift(-1).div(prices[tickers]).sub(1)
    baseline_scores = -prices[tickers].pct_change(
        periods=score_lookback, fill_method=None
    )
    daily_ic, ic_summary, bucket_summary = evaluate_score_buckets(
        baseline_scores,
        forward_returns,
        regimes,
        start_date=test_start,
        bucket_count=bucket_count,
    )
    return {
        "pair_screen": pair_screen,
        "pair_regime_results": pair_regime_results,
        "daily_ic": daily_ic,
        "ic_summary": ic_summary,
        "bucket_summary": bucket_summary,
        "regime_cutoffs": regime_cutoffs,
    }


def download_adjusted_closes(
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
    if raw.empty or "Close" not in raw:
        raise RuntimeError("Yahoo Finance returned no adjusted closing prices")
    close_prices = raw["Close"]
    if isinstance(close_prices, pd.Series):
        close_prices = close_prices.to_frame(name=tickers[0])
    missing = sorted(set(tickers) - set(close_prices.columns))
    if missing:
        raise RuntimeError(f"Yahoo Finance returned no prices for: {missing}")
    return close_prices.loc[:, tickers].dropna(how="all")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Discover inverse ticker pairs and evaluate a holdout baseline by volatility regime."
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        help="Explicit ticker list; defaults to the current official Nasdaq-100 roster.",
    )
    parser.add_argument("--benchmark", default="SPY")
    parser.add_argument("--start", default="2012-01-01")
    parser.add_argument("--end", default="2026-10-02", help="Exclusive end date")
    parser.add_argument("--train-end", default="2019-12-31")
    parser.add_argument("--test-start", default="2020-01-01")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/ticker_research_nasdaq100")
    )
    parser.add_argument("--relation-window", type=int, default=60)
    parser.add_argument("--regime-window", type=int, default=20)
    parser.add_argument("--score-lookback", type=int, default=20)
    parser.add_argument("--buckets", type=int, default=5)
    args = parser.parse_args()

    if args.tickers is None:
        universe_tickers, universe_as_of = fetch_nasdaq100_tickers()
    else:
        universe_tickers = list(dict.fromkeys(args.tickers))
        universe_as_of = "user-specified"
    tickers = list(dict.fromkeys([*universe_tickers, args.benchmark]))
    prices = download_adjusted_closes(tickers, start_date=args.start, end_date=args.end)
    results = analyze_prices(
        prices,
        benchmark=args.benchmark,
        train_end=args.train_end,
        test_start=args.test_start,
        relation_window=args.relation_window,
        regime_window=args.regime_window,
        score_lookback=args.score_lookback,
        bucket_count=args.buckets,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prices.to_csv(args.output_dir / "adjusted_closes.csv")
    pd.DataFrame(
        {"ticker": universe_tickers, "universe_as_of": universe_as_of}
    ).to_csv(args.output_dir / "universe.csv", index=False)
    for name, frame in results.items():
        if isinstance(frame, pd.DataFrame):
            frame.to_csv(args.output_dir / f"{name}.csv", index=False)

    candidates = results["pair_screen"].query("persistent_inverse")
    pair_regime_summary = results["pair_regime_results"]
    print(f"Saved results to {args.output_dir}")
    print(f"Tickers analyzed: {len(tickers) - 1}")
    print(f"Persistent inverse pairs found in training: {len(candidates)}")
    print(f"Train-fitted SPY volatility cutoffs: {results['regime_cutoffs']}")
    if not pair_regime_summary.empty:
        print("Holdout pair correlations by training-discovery and test regime:")
        print(
            pair_regime_summary.groupby(
                ["discovered_regime", "test_regime"], as_index=False
            )
            .agg(
                pairs=("test_correlation", "count"),
                train_candidates=("train_persistent_inverse", "sum"),
                mean_test_correlation=("test_correlation", "mean"),
            )
            .to_string(index=False)
        )
    print("Holdout mean rank IC by SPY volatility regime:")
    print(results["ic_summary"].to_string(index=False))
    print("Holdout mean next-day return by score bucket and regime:")
    print(results["bucket_summary"].to_string(index=False))


if __name__ == "__main__":
    main()