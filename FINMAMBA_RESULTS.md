# FinMamba Individual-Ticker Experiment: Preliminary Results

## Research question

Can the FinMamba per-ticker score rank next-five-day relative returns, and do its rankings behave differently across SPY volatility regimes? Separately, do individual tickers exhibit persistent inverse daily-return relationships in the training period?

## Data and target

- Ticker roster: the 101 Nasdaq-100 constituents returned by Nasdaq's API on 2026-10-02. Seventeen were excluded because they lacked complete OHLCV history over the selected period; the balanced model panel contains 84 tickers.
- Market data: adjusted daily OHLCV downloaded with `yfinance` and `auto_adjust=True`.
- Model dates: 2018-01-03 through 2026-09-24, 2,193 trading dates.
- Features: open, high, low, and close returns relative to prior close, plus log volume. Per-ticker feature scaling uses training dates only.
- Label: each ticker's next-five-trading-day adjusted return minus the same-date universe mean.
- Split: train through 2021-12-31, validation during 2022, holdout from 2023-01-01 through 2026-09-24.
- The industry-relation matrix is all ones because point-in-time industry classifications were not obtained; the graph therefore uses daily correlations without a sector mask.

## Training run

- One epoch, seed 2024, batch size 16, learning rate 0.001.
- Small model configuration: GAT hidden channels 16, two Mamba branches with hidden size 32, state size 16.
- Runtime: Docker Linux container, CPU execution, PyTorch 2.8.0. The Mamba kernel uses the upstream pure-PyTorch reference recurrence (`FINMAMBA_MAMBA_REFERENCE=1`) because this WSL host has no NVIDIA adapters. This is much slower than the optimized CUDA kernel and is a feasibility run, not a full reproduction of the paper's configuration.
- The training entry point produced a checkpoint, test scores, and predictions under `outputs/finmamba_reference/`.

## Holdout results

SPY volatility is 20-day annualized realized volatility. Tercile thresholds are fitted using model training dates only: low cutoff 10.38%, high cutoff 17.15%.

| SPY volatility regime | Holdout days | FinMamba mean rank IC | FinMamba median rank IC | Mean return, low-score bucket | Mean return, high-score bucket |
|---|---:|---:|---:|---:|---:|
| High | 126 | -0.0010 | -0.0147 | +0.073% | -0.016% |
| Low | 263 | -0.0351 | -0.0318 | +0.163% | -0.089% |
| Mid | 546 | -0.0076 | -0.0005 | +0.170% | +0.018% |

The top-score bucket underperformed the bottom-score bucket in all three regimes. This one-epoch run does not show useful ranking performance. The 20-day momentum and mean-reversion baselines are also saved in `finmamba_ic_summary.csv` and `finmamba_score_buckets.csv` for comparison; they are diagnostics, not cost-adjusted strategy results.

## Inverse-pair results

Pairs are discovered only from 2018-2021 returns, separately within train-fitted SPY volatility regimes. A persistent inverse pair must have a latest 60-return window correlation at or below -0.3 and meet that threshold in at least 75% of its observed training windows.

- Persistent inverse ticker/regime candidates: 0.
- Mean holdout pairwise correlation across the 84-ticker universe: +0.398 in high, +0.106 in low, and +0.170 in mid SPY-volatility regimes.

This sample does not support a stable inverse-pair long-short rule under the chosen thresholds. Correlation is descriptive and does not predict which ticker will fall next.

## Implementation observations and limitations

- In a backward-pass smoke test, the four `MarketGuideInception` parameters had no gradients. The forward path converts its learned rate to a Python integer for top-k selection, which disconnects the guide from the loss. This should be addressed as a separate architecture experiment, not hidden in these baseline results.
- Today's Nasdaq-100 members applied to earlier dates create survivorship bias; the 84-ticker panel is not point-in-time constituent data.
- One training epoch and the reduced model configuration are only a feasibility baseline. No transaction costs, borrow fees, slippage, position sizing, or short availability are modeled.
- Do not interpret these results as a trading recommendation or as evidence that FinMamba has been fully reproduced.
