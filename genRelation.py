from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm


_COLUMN_ALIASES = {
    "kdcode": "instrument",
    "dt": "datetime",
    "Company": "instrument",
    "Date": "datetime",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate one dynamic stock-relation matrix for each trading day."
    )
    parser.add_argument("--stock", type=str, default="nasdaq")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <stock>_stock_relation.",
    )
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument(
        "--method",
        choices=("spearman", "pcc"),
        default="spearman",
    )
    parser.add_argument("--start-day", type=int, default=0)
    parser.add_argument(
        "--end-day",
        type=int,
        default=None,
        help="Exclusive end index; defaults to the total number of days.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto, cpu, cuda, cuda:0, cuda:1, ...",
    )
    return parser


def resolve_device(specification: str) -> torch.device:
    if specification == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    device = torch.device(specification)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {device}")
        index = 0 if device.index is None else device.index
        if index >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device {device} does not exist; found {torch.cuda.device_count()} device(s)."
            )
    return device


def load_feature_tensor(
    *, stock: str, data_dir: Path, device: torch.device
) -> tuple[torch.Tensor, int, int, int]:
    feature_path = data_dir / f"{stock}fea.pkl"
    if not feature_path.exists():
        raise FileNotFoundError(f"Feature file not found: {feature_path}")

    frame = pd.read_pickle(feature_path).reset_index()
    frame = frame.rename(columns=_COLUMN_ALIASES)
    missing = {"instrument", "datetime"} - set(frame.columns)
    if missing:
        raise ValueError(
            "Feature data must contain instrument/datetime columns (or supported aliases); "
            f"missing: {sorted(missing)}"
        )

    stock_num = int(frame["instrument"].nunique())
    day_num = int(frame["datetime"].nunique())
    feature_num = int(frame.shape[1] - 2)
    values = frame.drop(columns=["datetime", "instrument"]).to_numpy(copy=True)
    expected_shape = (day_num * stock_num, feature_num)
    if values.shape != expected_shape:
        raise ValueError(
            f"Feature values have shape {values.shape}, expected {expected_shape}. "
            "Check that every day contains the same stocks and that rows follow the "
            "same ordering used by the training code."
        )

    tensor = torch.as_tensor(
        values.reshape(day_num, stock_num, feature_num),
        dtype=torch.float32,
        device=device,
    )
    return tensor, stock_num, day_num, feature_num


def cal_pccs(x: torch.Tensor, y: torch.Tensor, n: int) -> torch.Tensor:
    sum_xy = torch.sum(x * y)
    sum_x = torch.sum(x)
    sum_y = torch.sum(y)
    sum_x2 = torch.sum(x * x)
    sum_y2 = torch.sum(y * y)
    return (n * sum_xy - sum_x * sum_y) / torch.sqrt(
        (n * sum_x2 - sum_x * sum_x)
        * (n * sum_y2 - sum_y * sum_y)
        + 1e-6
    )


def cal_spearman(x: torch.Tensor, y: torch.Tensor, n: int) -> torch.Tensor:
    rank_x = torch.argsort(torch.argsort(x))
    rank_y = torch.argsort(torch.argsort(y))

    sum_rank_xy = torch.sum(rank_x * rank_y)
    sum_rank_x = torch.sum(rank_x)
    sum_rank_y = torch.sum(rank_y)
    sum_rank_x2 = torch.sum(rank_x * rank_x)
    sum_rank_y2 = torch.sum(rank_y * rank_y)
    return (n * sum_rank_xy - sum_rank_x * sum_rank_y) / torch.sqrt(
        (n * sum_rank_x2 - sum_rank_x * sum_rank_x)
        * (n * sum_rank_y2 - sum_rank_y * sum_rank_y)
        + 1e-6
    )


def calculate_relations(
    xs: torch.Tensor,
    all_stocks: torch.Tensor,
    *,
    n: int,
    method: str,
) -> torch.Tensor:
    if xs.ndim != 2 or all_stocks.ndim != 3:
        raise ValueError("xs must have shape (features, days) and all_stocks (stocks, features, days)")
    if xs.shape != all_stocks.shape[1:]:
        raise ValueError(
            f"xs shape {tuple(xs.shape)} must match each stock panel {tuple(all_stocks.shape[1:])}"
        )
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")

    if method == "spearman":
        target = torch.argsort(torch.argsort(xs, dim=-1), dim=-1).to(all_stocks.dtype)
        values = torch.argsort(
            torch.argsort(all_stocks, dim=-1), dim=-1
        ).to(all_stocks.dtype)
    elif method == "pcc":
        target = xs
        values = all_stocks
    else:
        raise ValueError(f"Unknown correlation method: {method!r}")

    target_sum = target.sum(dim=-1)
    target_sum_squared = target.square().sum(dim=-1)
    values_sum = values.sum(dim=-1)
    values_sum_squared = values.square().sum(dim=-1)
    sum_products = torch.einsum("ft,nft->nf", target, values)

    numerator = n * sum_products - target_sum.unsqueeze(0) * values_sum
    target_variance = n * target_sum_squared - target_sum.square()
    values_variance = n * values_sum_squared - values_sum.square()
    denominator = torch.sqrt(values_variance * target_variance.unsqueeze(0) + 1e-6)
    return torch.nan_to_num((numerator / denominator).mean(dim=1), nan=0.0)


def stock_cor_matrix(
    features: torch.Tensor,
    *,
    lookback: int,
    day: int,
    method: str,
) -> torch.Tensor:
    start_day = max(0, day - (lookback - 1))
    window = features[start_day : day + 1].permute(1, 2, 0)
    stock_num = window.size(0)
    if method == "spearman":
        values = torch.argsort(
            torch.argsort(window, dim=-1), dim=-1
        ).to(features.dtype)
    elif method == "pcc":
        values = window
    else:
        raise ValueError(f"Unknown correlation method: {method!r}")

    sum_values = values.sum(dim=-1).transpose(0, 1)
    sum_values_squared = values.square().sum(dim=-1).transpose(0, 1)
    feature_values = values.permute(1, 0, 2)
    sum_products = torch.einsum("fit,fjt->fij", feature_values, feature_values)
    numerator = lookback * sum_products - sum_values.unsqueeze(2) * sum_values.unsqueeze(1)
    variance = lookback * sum_values_squared - sum_values.square()
    denominator = torch.sqrt(variance.unsqueeze(2) * variance.unsqueeze(1) + 1e-6)
    relation = torch.nan_to_num(
        (numerator / denominator).mean(dim=0),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    relation.fill_diagonal_(1)
    return relation


def main() -> None:
    args = build_parser().parse_args()
    if args.lookback <= 0:
        raise ValueError(f"--lookback must be positive, got {args.lookback}")

    device = resolve_device(args.device)
    features, stock_num, day_num, feature_num = load_feature_tensor(
        stock=args.stock,
        data_dir=args.data_dir,
        device=device,
    )
    print(
        f"device={device}, tensor_shape={tuple(features.shape)}, "
        f"stocks={stock_num}, days={day_num}, features={feature_num}"
    )

    start_day = args.start_day
    end_day = day_num if args.end_day is None else args.end_day
    if not 0 <= start_day <= end_day <= day_num:
        raise ValueError(
            f"Expected 0 <= start-day <= end-day <= {day_num}, got "
            f"start-day={start_day}, end-day={end_day}."
        )

    output_dir = (
        Path(f"{args.stock}_stock_relation")
        if args.output_dir is None
        else args.output_dir
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    for day in tqdm(range(start_day, end_day), desc="relations"):
        relation = stock_cor_matrix(
            features,
            lookback=args.lookback,
            day=day,
            method=args.method,
        )
        with (output_dir / f"day{day}.pkl").open("wb") as handle:
            pickle.dump(relation.detach().cpu().numpy(), handle)

    print(f"saved {end_day - start_day} relation matrix/matrices to {output_dir}")


if __name__ == "__main__":
    main()
