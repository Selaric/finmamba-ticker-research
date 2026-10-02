<p align="center">
    <img src="./assets/logo.png" width="150">
</p>

# FinMamba: Market-Aware Graph Enhanced Multi-Level Mamba for Stock Movement Prediction

## 📰 News

🚩 2026-06-22: FinMamba has been accepted as KDD 2026 Workshop on Machine Learning in Finance Oral.

🚩 2025-02-10: Initial upload to arXiv [PDF](https://arxiv.org/abs/2502.06707).

## 🌟 Overview

FinMamba consists of  Temporal Stock-Correlation Graph learning and Multi-Scale Mamba for stock movement prediction.

![](./assets/FinMamba.png)

## 🛠 Usage

The complete FinMamba workflow consists of three sequential stages:

```text
Generate daily relation graphs → Train FinMamba → Run the backtest
```

Please execute the steps below in this order. In particular, the daily short-term stock-relation graphs must be generated before training starts.

### Prerequisites

Install the required Python packages:

```bash
pip install -r requirements.txt
```

To run the backtest notebook, also make sure that Jupyter Notebook is available in the environment:

```bash
pip install notebook
```

For the NASDAQ 100 example, the expected input files are organized as follows:

```text
data/
├── nasdaqfea.pkl
├── nasdaqlab.pkl
└── nasdaq_industry_relationship.npy

stock/
└── NASDAQ100_new.csv
```

The feature data, label data, industry-relation matrix, and backtest market data must use a consistent stock universe and trading calendar.

### Step 1: Generate Daily Short-Term Stock-Relation Graphs

Before training, run `genRelation.py` to generate a short-term relation graph for every trading day. For each day, the script uses a rolling historical window ending on that day to estimate pairwise stock correlations and saves the resulting adjacency matrix as a pickle file.

For example, the following command generates 20-day Spearman relation graphs for the NASDAQ 100 universe:

```bash
python genRelation.py \
  --stock nasdaq \
  --data-dir data \
  --output-dir nasdaq_stock_relation \
  --lookback 20 \
  --method spearman \
  --device auto
```

The generated files are stored as:

```text
nasdaq_stock_relation/
├── day0.pkl
├── day1.pkl
├── day2.pkl
└── ...
```

One `day{index}.pkl` file is required for each trading day used by the training, validation, and test periods. The output directory must be the same as the relation directory configured for training. Use `--method pcc` instead of `--method spearman` to generate Pearson-correlation graphs.

### Analyze Inverse Relationships Between Individual Tickers

The standalone research baseline downloads adjusted closes, screens individual ticker pairs for persistent negative daily-return correlation using training data only, labels low/mid/high SPY-volatility regimes using training-period terciles, and evaluates a simple 20-day mean-reversion score by daily Spearman rank IC and forward-return buckets. It does not modify or train FinMamba; it gives us a transparent baseline to compare against.

By default, the script fetches the current 101-symbol Nasdaq-100 roster from Nasdaq's components API and saves the roster date. Applying today's members to historical years creates survivorship bias; use point-in-time index membership before treating results as robust. A smaller explicit universe can be supplied with `--tickers`.

Set up the project-local environment in PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-research.txt
.\.venv\Scripts\python.exe research_ticker_pairs.py
```

The default training period ends on 2019-12-31 and the holdout begins on 2020-01-01. The pair screen uses non-overlapping 60-return windows; a candidate must have a latest training correlation at or below -0.3 and meet that threshold in at least 75% of its observed windows. The SPY 20-day realized-volatility cutoffs are fit on training data and then held fixed in the test period. The baseline score is the negative 20-day past return; its daily rank IC and next-day-return buckets are evaluated only on the holdout.

The script writes adjusted closes, the roster snapshot, training-selected pairs, pair correlations by test regime, daily and summary IC, and return buckets under `outputs/ticker_research_nasdaq100/`. Pair correlation is descriptive, not a prediction that one ticker will fall when another rises. The score is a deliberately simple baseline, not a FinMamba result or a trading signal; include costs and realistic shorting constraints in any later strategy test.

See [PAIR_RESEARCH_RESULTS.md](PAIR_RESEARCH_RESULTS.md) for the pair-screen baseline and [FINMAMBA_RESULTS.md](FINMAMBA_RESULTS.md) for the trained-model experiment.

### CPU Reference Training

For a Linux CPU run without an NVIDIA adapter, build the isolated image and prepare the OHLCV panels and daily relation files first:

```powershell
docker build -t finmamba-cpu -f Dockerfile.training .
.\.venv\Scripts\python.exe prepare_finmamba_data.py
docker run --rm -v "${PWD}:/workspace" -w /workspace finmamba-cpu python genRelation.py --stock nasdaq --data-dir data --output-dir nasdaq_stock_relation --lookback 20 --method spearman --device cpu
```

Then train and evaluate:

```powershell
docker run --rm -v "${PWD}:/workspace" -w /workspace finmamba-cpu python train_finmamba.py --stock nasdaq --data-dir data --relation-dir nasdaq_stock_relation --train-start 2018-01-01 --train-end 2021-12-31 --valid-start 2022-01-01 --valid-end 2022-12-31 --test-start 2023-01-01 --test-end 2026-09-24 --seq-len 20 --epochs 1 --batch-size 16 --device cpu --output-dir outputs/finmamba_reference --prediction-layout date-major
docker run --rm -v "${PWD}:/workspace" -w /workspace finmamba-cpu python evaluate_finmamba_outputs.py
```

The training image sets `FINMAMBA_MAMBA_REFERENCE=1`, which uses the upstream PyTorch reference scan and disables the optimized CUDA path. It preserves the Mamba recurrence but is slower and is intended for small feasibility experiments. The standard path remains unchanged when this environment variable is unset.

### Step 2: Train and Evaluate FinMamba

After all daily relation graphs have been generated, train FinMamba with:

```bash
bash run_finmamba.sh
```

All model, optimization, data-split, device, and output parameters are defined in `run_finmamba.sh`. By default, the script reads the relation graphs from `nasdaq_stock_relation/` and produces:

```text
best_model.pth   # best checkpoint selected on the validation set
scores.csv       # raw prediction scores on the test set
pred.csv         # predictions aligned with stock identifiers and dates
```

Common paths and the device can be overridden without editing the script. For example:

```bash
DEVICE=cuda:1 \
RELATION_DIR=nasdaq_stock_relation \
OUTPUT_DIR=outputs/nasdaq \
bash run_finmamba.sh
```

### Step 3: Run the Backtest

After training is complete, use `backtest.ipynb` to evaluate the trading performance of the predictions in `pred.csv`.

By default, the notebook reads:

```text
pred.csv
stock/NASDAQ100_new.csv
```

Start Jupyter Notebook from the project root:

```bash
jupyter notebook backtest.ipynb
```

Then run the notebook cells in order to construct the prediction-based portfolio and calculate the backtest results. If `OUTPUT_DIR` was changed during training, update the `pred.csv` path in the notebook accordingly. For another stock universe, also update the market-data path and related date settings in the notebook.

## 📊 Dataset

### Form
The provided data is split into training, validation, and test sets, with 4 stock universes. (CSI 300, CSI 500, NASDAQ 100, S&P 500)

In our code, the data will be gathered chronically and then grouped by prediction dates. the data iterated by the data loader is of shape (T, N, F), where:

- T - length of lookback_window, T=20.
- N - number of stocks. 
- F - 5 for NASDAQ 100 and S&P 500 including high, low, open, close, volume. 6 for CSI 300 and CSI 500 including high, low, open, close, volume, turnover.

### Market Index
For convenience and fairness, we extract the market index as the mean value of the stocks included in the target set.

## 📚 Citation

If you find this repository helpful, please cite our paper:

```bibtex
@article{hu2025finmamba,
  title={Finmamba: Market-aware graph enhanced multi-level mamba for stock movement prediction},
  author={Hu, Yifan and Liu, Peiyuan and Li, Yuante and Cheng, Dawei and Li, Naiqi and Dai, Tao and Bao, Jigang and Xia, Shu-Tao},
  journal={arXiv preprint arXiv:2502.06707},
  year={2025}
}
```



