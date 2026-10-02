# FinMamba Ticker Research

An experimental research project built on the upstream FinMamba implementation. We evaluate whether the model can rank individual stocks by future relative returns, test for persistent inverse relationships between ticker pairs, and measure performance across SPY volatility regimes.

This is research software, not a trading system or investment advice.

## What We Built

- An individual-ticker inverse-return screen using adjusted daily prices and train-only regime thresholds.
- A data-preparation pipeline for balanced Nasdaq-100 OHLCV panels and five-day excess-return labels.
- A corrected, vectorized daily stock-relation generator, with regression tests against pairwise Pearson and Spearman calculations.
- An opt-in CPU reference path for Mamba, plus an isolated Linux Docker image.
- Holdout evaluation for rank information coefficient (IC), score buckets, volatility regimes, and ticker-pair stability.

The model outputs one score per ticker. It does not directly select long-short pairs. Pair discovery is analyzed separately, and the current graph construction does not preserve inverse correlations as signed relationships.

## Experiment

We used the Nasdaq-100 roster returned by Nasdaq's API on 2026-10-02. Of 101 current constituents, 84 had complete adjusted OHLCV data from 2018-01-03 through 2026-09-24 and were included in the balanced model panel. Applying today's roster to historical data introduces survivorship bias.

The model input contains five features per ticker: open, high, low, and close returns relative to the prior close, plus log volume. Features are standardized per ticker using training-period statistics only. The target is each ticker's next-five-trading-day adjusted return minus the same-day universe mean.

| Split | Dates |
|---|---|
| Training | 2018 through 2021 |
| Validation | 2022 |
| Holdout | 2023-01-01 through 2026-09-24 |

The industry-relation matrix was set to all ones because point-in-time industry classifications were unavailable. The graph therefore uses daily stock correlations without an industry mask.

## Results

The run used one epoch and a reduced model configuration in CPU reference mode. It is a feasibility baseline, not a full reproduction of the paper.

| SPY volatility regime | Holdout days | Mean rank IC | Median rank IC | Lowest-score bucket return | Highest-score bucket return |
|---|---:|---:|---:|---:|---:|
| High | 126 | -0.0010 | -0.0147 | +0.073% | -0.016% |
| Low | 263 | -0.0351 | -0.0318 | +0.163% | -0.089% |
| Mid | 546 | -0.0076 | -0.0005 | +0.170% | +0.018% |

The score buckets contain mean next-five-day excess returns. FinMamba's highest-score bucket underperformed its lowest-score bucket in every regime in this run. These results do not show useful ranking performance.

The inverse-pair screen found **0 persistent inverse pairs** among 10,458 ticker-pair/regime rows. The rule required a latest 60-return training-window correlation at or below -0.3, with at least 75% of training windows meeting that threshold. Mean pairwise holdout correlation was +0.398 in high-, +0.106 in low-, and +0.170 in mid-SPY-volatility periods.

SPY's 20-day annualized realized volatility defines low, mid, and high regimes. Tercile thresholds were fit using training data only: 10.38% and 17.15%.

See [FINMAMBA_RESULTS.md](FINMAMBA_RESULTS.md) for full methodology, baselines, implementation notes, and caveats. See [PAIR_RESEARCH_RESULTS.md](PAIR_RESEARCH_RESULTS.md) for the detailed pair-screen results.

## Reproduce

### Local Research Environment

From the repository root in PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-research.txt
.\.venv\Scripts\python.exe research_ticker_pairs.py
.\.venv\Scripts\python.exe prepare_finmamba_data.py
```

The data-preparation script downloads the current Nasdaq-100 roster and adjusted OHLCV. Yahoo Finance data and derived files are not committed to this repository.

### CPU Reference Run

Docker Desktop with its Linux engine is required. The image uses a CUDA-enabled PyTorch runtime because of the Mamba package's build requirements, but the commands below run on CPU and do not require a host NVIDIA GPU. The reference scan is much slower than the optimized CUDA kernel.

```powershell
docker build -t finmamba-cpu -f Dockerfile.training .
$workspace = (Get-Location).Path
docker run --rm -v "${workspace}:/workspace" -w /workspace finmamba-cpu python genRelation.py --stock nasdaq --data-dir data --output-dir nasdaq_stock_relation --lookback 20 --method spearman --device cpu
docker run --rm -v "${workspace}:/workspace" -w /workspace finmamba-cpu python train_finmamba.py --stock nasdaq --data-dir data --relation-dir nasdaq_stock_relation --relation-pattern 'day{index}.pkl' --train-start 2018-01-01 --train-end 2021-12-31 --valid-start 2022-01-01 --valid-end 2022-12-31 --test-start 2023-01-01 --test-end 2026-09-24 --seq-len 20 --market-kernel-sizes 4 10 20 --gat-hidden-channels 16 --gat-out-channels auto --gat-layers 2 --gat-heads 2 --mamba-hidden-sizes 32 32 --mamba-num-heads 2 --mamba-output-size 16 --mamba-d-state 16 --mamba-d-conv 2 --mamba-expand 1 --dropout 0.1 --epochs 1 --batch-size 16 --learning-rate 0.001 --weight-decay 1e-7 --patience 1 --log-interval 20 --seed 2024 --device cpu --output-dir outputs/finmamba_reference --prediction-layout date-major
docker run --rm -v "${workspace}:/workspace" -w /workspace finmamba-cpu python evaluate_finmamba_outputs.py --train-start 2018-01-01 --train-end 2021-12-31 --test-start 2023-01-01
```

The Docker image sets `FINMAMBA_MAMBA_REFERENCE=1`, which selects the upstream pure-PyTorch reference recurrence and disables the optimized CUDA path. Unset that variable in a CUDA development environment to use the normal fast path.

## Important Limitations

- The Nasdaq-100 roster is current, not point-in-time; historical membership and survivorship bias are not controlled.
- Yahoo Finance is the data source; no transaction costs, slippage, borrow fees, short availability, or portfolio sizing are modeled.
- The model experiment uses one epoch, a reduced configuration, and CPU reference execution. Its results are preliminary and must not be compared directly with the paper's reported results.
- The current `MarketGuideInception` parameters receive no gradients in the tested backward pass because the learned rate is converted to an integer for top-k selection. This is a model-design issue to investigate before further tuning.
- Pairwise correlation is descriptive and does not establish that one ticker will fall when another rises.

Generated prices, relation matrices, checkpoints, and CSV outputs are excluded from Git by `.gitignore`.

## Upstream Credit

This project is derived from [TROUBADOUR000/FinMamba](https://github.com/TROUBADOUR000/FinMamba), the PyTorch implementation of **FinMamba: Market-Aware Graph Enhanced Multi-Level Mamba for Stock Movement Prediction** by Yifan Hu, Peiyuan Liu, Yuante Li, Dawei Cheng, Naiqi Li, Tao Dai, Jigang Bao, and Shu-Tao Xia. The original repository reports acceptance to the KDD 2026 Workshop on Machine Learning in Finance.

The upstream model and original assets are retained where used. This repository's ticker-pair analysis, data preparation, relation-generation correction, CPU reference option, evaluation, tests, and experiment reports are our additions. The upstream `LICENSE` is retained; see it for the Apache License 2.0 terms.

Paper: [arXiv:2502.06707](https://arxiv.org/abs/2502.06707)

```bibtex
@article{hu2025finmamba,
  title={Finmamba: Market-aware graph enhanced multi-level mamba for stock movement prediction},
  author={Hu, Yifan and Liu, Peiyuan and Li, Yuante and Cheng, Dawei and Li, Naiqi and Dai, Tao and Bao, Jigang and Xia, Shu-Tao},
  journal={arXiv preprint arXiv:2502.06707},
  year={2025}
}
```
