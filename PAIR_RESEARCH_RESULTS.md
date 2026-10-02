# Individual Ticker Inverse-Pair Results

## Question

Do individual ticker pairs have persistent negative return correlations within SPY volatility regimes, using the same universe and training dates as the FinMamba experiment?

## Data and method

- Model universe: 84 tickers with complete OHLCV data, selected from the current 101-member Nasdaq-100 roster returned by Nasdaq's API on 2026-10-02. The roster is not point-in-time, so historical results have survivorship bias.
- Adjusted daily closes downloaded from Yahoo Finance with `auto_adjust=True`.
- Price data spans 2018-01-03 through 2026-09-24. Training ends 2021-12-31; holdout starts 2023-01-01. 2022 is reserved for validation.
- SPY 20-day annualized realized-volatility terciles are fit on training dates only, then held fixed in the holdout. Cutoffs are 10.38% and 17.15%.
- Each regime's training correlations use non-overlapping 60-return windows. A pair is persistently inverse only if its latest training-window correlation is at most -0.3 and at least 75% of its windows meet that same threshold.

## Results

None of the 10,458 ticker-pair/training-regime rows passed the persistence rule.

| Training regime | Most negative mean train correlation | Pair | Latest train-window correlation | Fraction of windows at or below -0.3 |
|---|---:|---|---:|---:|
| High | +0.006 | FANG / FER | +0.092 | 0.00 |
| Low | -0.245 | FANG / XEL | -0.189 | 0.20 |
| Mid | -0.141 | AEP / STX | +0.002 | 0.20 |

Mean pair correlation over the holdout:

| Holdout SPY-volatility regime | Mean correlation | Median correlation |
|---|---:|---:|
| High | +0.398 | +0.410 |
| Low | +0.106 | +0.093 |
| Mid | +0.170 | +0.155 |

## Interpretation and limits

Under these thresholds, this 84-ticker sample provides no evidence of a persistent inverse pair suitable for a long-short rule. The relatively weak negative training correlations did not persist consistently. Correlation is descriptive; it does not predict which ticker will fall next.

This is a research screen, not a trading result. The roster has survivorship bias, Yahoo data is not point-in-time index membership, and the analysis excludes transaction costs, borrow availability, short rebates, slippage, and portfolio sizing. The downloaded inputs and CSV outputs remain local and are ignored by Git.
