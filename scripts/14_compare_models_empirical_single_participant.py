"""Compare SBA and WBA for empirical participant 201 using pointwise log_lik.

This script reuses the saved single participant Stan outputs when available.
It extracts pointwise log likelihood values.
It computes simple comparison measures.
It saves a compact comparison table.

Run from the project root:
    python scripts/14_compare_models_empirical_single_participant.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PARTICIPANT_ID = 201


def ensure_directories(*paths: Path) -> None:
    """Create output directories if they do not already exist."""
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def load_cmdstan_chain_csvs(paths: list[Path]) -> pd.DataFrame:
    """Load and concatenate CmdStan chain CSV files."""
    if not paths:
        raise FileNotFoundError("No CmdStan chain CSV files were provided.")
    chain_frames = [pd.read_csv(path, comment="#") for path in paths]
    return pd.concat(chain_frames, ignore_index=True)


def get_latest_chain_paths(results_dir: Path, model_prefix: str) -> list[Path]:
    """Find the latest saved four chain CmdStan output set for a model prefix."""
    chain_files = sorted(results_dir.glob(f"{model_prefix}-*_*.csv"))
    if not chain_files:
        raise FileNotFoundError(f"No chain CSV files found for model prefix '{model_prefix}' in {results_dir}")

    grouped_paths: dict[str, list[Path]] = {}
    for path in chain_files:
        stem = path.stem
        prefix, chain_suffix = stem.rsplit("_", 1)
        if chain_suffix.isdigit():
            grouped_paths.setdefault(prefix, []).append(path)

    if not grouped_paths:
        raise FileNotFoundError(f"Could not group chain CSV files for model prefix '{model_prefix}'.")

    latest_prefix = max(grouped_paths.keys(), key=lambda item: item.split("-")[-1])
    latest_paths = sorted(grouped_paths[latest_prefix], key=lambda path: int(path.stem.rsplit("_", 1)[1]))
    return latest_paths


def extract_log_lik_matrix(draws_df: pd.DataFrame) -> np.ndarray:
    """Extract the pointwise log likelihood matrix from concatenated draws."""
    log_lik_columns = [column for column in draws_df.columns if column.startswith("log_lik.")]
    if not log_lik_columns:
        raise ValueError("No pointwise log_lik columns were found in the saved Stan draws.")

    log_lik_columns = sorted(log_lik_columns, key=lambda name: int(name.split(".")[1]))
    return draws_df[log_lik_columns].to_numpy(dtype=float)


def compute_basic_metrics(log_lik_matrix: np.ndarray) -> dict[str, float]:
    """Compute total and mean pointwise log likelihood."""
    pointwise_log_lik = log_lik_matrix.mean(axis=0)
    total_log_lik = float(pointwise_log_lik.sum())
    mean_log_lik = float(pointwise_log_lik.mean())
    return {
        "total_log_lik": total_log_lik,
        "mean_log_lik": mean_log_lik,
    }


def compute_waic(log_lik_matrix: np.ndarray) -> dict[str, float]:
    """Compute WAIC from pointwise log likelihood draws."""
    likelihood_matrix = np.exp(log_lik_matrix)
    lppd = np.sum(np.log(np.mean(likelihood_matrix, axis=0)))
    p_waic = np.sum(np.var(log_lik_matrix, axis=0, ddof=1))
    waic = -2.0 * (lppd - p_waic)
    return {
        "waic": float(waic),
        "elpd_waic": float(lppd - p_waic),
    }


def build_model_row(
    participant_id: int,
    model_name: str,
    chain_paths: list[Path],
) -> dict:
    """Build one model comparison row from saved chain CSV files."""
    draws_df = load_cmdstan_chain_csvs(chain_paths)
    log_lik_matrix = extract_log_lik_matrix(draws_df)
    basic_metrics = compute_basic_metrics(log_lik_matrix)
    waic_metrics = compute_waic(log_lik_matrix)

    row = {
        "participant_id": participant_id,
        "model": model_name,
        "total_log_lik": basic_metrics["total_log_lik"],
        "mean_log_lik": basic_metrics["mean_log_lik"],
        "WAIC": waic_metrics["waic"],
        "elpd_waic": waic_metrics["elpd_waic"],
    }
    return row


def print_summary(df: pd.DataFrame) -> None:
    """Print the comparison table and a short textual interpretation."""
    display_df = df.copy()
    numeric_columns = ["total_log_lik", "mean_log_lik", "WAIC", "elpd_waic"]
    display_df[numeric_columns] = display_df[numeric_columns].round(3)

    print("=" * 90)
    print("Empirical single-participant model comparison")
    print("=" * 90)
    print(display_df.to_string(index=False))

    best_by_waic = df.sort_values("WAIC").iloc[0]
    best_by_loglik = df.sort_values("total_log_lik", ascending=False).iloc[0]

    print("\nBrief summary:")
    print(
        f"- By WAIC, the better model for participant {PARTICIPANT_ID} is {best_by_waic['model']} "
        f"(lower is better: {best_by_waic['WAIC']:.3f})."
    )
    print(
        f"- By total log likelihood, the better model for participant {PARTICIPANT_ID} is "
        f"{best_by_loglik['model']} (higher is better: {best_by_loglik['total_log_lik']:.3f})."
    )


def main() -> None:
    """Reuse saved single participant fits and compare SBA vs WBA for participant 201."""
    project_root = Path(__file__).resolve().parents[1]
    empirical_dir = project_root / "results" / "fits" / "empirical"
    ensure_directories(empirical_dir)

    wba_chain_paths = get_latest_chain_paths(empirical_dir, "wba_model")
    sba_chain_paths = get_latest_chain_paths(empirical_dir, "sba_model")

    comparison_rows = [
        build_model_row(PARTICIPANT_ID, "WBA", wba_chain_paths),
        build_model_row(PARTICIPANT_ID, "SBA", sba_chain_paths),
    ]
    comparison_df = pd.DataFrame(comparison_rows)

    output_path = empirical_dir / "empirical_single_participant_model_comparison.csv"
    comparison_df.to_csv(output_path, index=False)

    print_summary(comparison_df)
    print("\nSaved comparison table to:")
    print(output_path)


if __name__ == "__main__":
    main()
