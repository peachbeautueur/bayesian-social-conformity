"""Perform empirical participant level model comparison for SBA vs WBA.

This script reuses saved no pooling empirical Stan outputs.
It extracts pointwise log likelihood values for each participant and model.
It computes comparison measures.
It saves a participant level comparison table.
It creates a small set of figures.

Run from the project root:
    python scripts/15_compare_models_empirical_all_participants.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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


def build_chain_groups(per_participant_dir: Path, model_prefix: str) -> list[tuple[str, list[Path], float]]:
    """Group chain CSV files by CmdStan timestamp prefix."""
    chain_files = sorted(per_participant_dir.glob(f"{model_prefix}-*_*.csv"))

    grouped_paths: dict[str, list[Path]] = {}
    grouped_times: dict[str, float] = {}
    for path in chain_files:
        stem = path.stem
        prefix, chain_suffix = stem.rsplit("_", 1)
        if not chain_suffix.isdigit():
            continue

        grouped_paths.setdefault(prefix, []).append(path)
        grouped_times[prefix] = max(grouped_times.get(prefix, 0.0), path.stat().st_mtime)

    groups = []
    for prefix, paths in grouped_paths.items():
        sorted_paths = sorted(paths, key=lambda item: int(item.stem.rsplit("_", 1)[1]))
        groups.append((prefix, sorted_paths, grouped_times[prefix]))

    return sorted(groups, key=lambda item: item[2])


def get_summary_stem_prefix(model_prefix: str) -> str:
    """Convert a Stan executable prefix such as `wba_model` into `wba`."""
    if model_prefix.endswith("_model"):
        return model_prefix[: -len("_model")]
    return model_prefix


def build_participant_chain_map(per_participant_dir: Path, model_prefix: str) -> dict[int, list[Path]]:
    """Map participant IDs to their saved chain CSV files using write order."""
    summary_stem_prefix = get_summary_stem_prefix(model_prefix)
    summary_paths = sorted(
        per_participant_dir.glob(f"{summary_stem_prefix}_empirical_participant_*_full_summary.csv"),
        key=lambda path: path.stat().st_mtime,
    )
    chain_groups = build_chain_groups(per_participant_dir, model_prefix)

    if len(summary_paths) != len(chain_groups):
        raise ValueError(
            f"Mismatch between summary files ({len(summary_paths)}) and chain groups ({len(chain_groups)}) "
            f"in {per_participant_dir}."
        )

    participant_map: dict[int, list[Path]] = {}
    for summary_path, (_, chain_paths, _) in zip(summary_paths, chain_groups):
        stem_parts = summary_path.stem.split("_")
        participant_index = stem_parts.index("participant")
        participant_id = int(stem_parts[participant_index + 1])
        participant_map[participant_id] = chain_paths

    return participant_map


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
    """Compute WAIC and elpd_waic from pointwise log likelihood draws."""
    likelihood_matrix = np.exp(log_lik_matrix)
    lppd = np.sum(np.log(np.mean(likelihood_matrix, axis=0)))
    p_waic = np.sum(np.var(log_lik_matrix, axis=0, ddof=1))
    waic = -2.0 * (lppd - p_waic)
    return {
        "WAIC": float(waic),
        "elpd_waic": float(lppd - p_waic),
    }


def compute_model_metrics(chain_paths: list[Path]) -> dict[str, float]:
    """Compute model comparison metrics from saved chain CSV files."""
    draws_df = load_cmdstan_chain_csvs(chain_paths)
    log_lik_matrix = extract_log_lik_matrix(draws_df)
    basic_metrics = compute_basic_metrics(log_lik_matrix)
    waic_metrics = compute_waic(log_lik_matrix)

    return {
        "total_log_lik": basic_metrics["total_log_lik"],
        "mean_log_lik": basic_metrics["mean_log_lik"],
        "WAIC": waic_metrics["WAIC"],
        "elpd_waic": waic_metrics["elpd_waic"],
    }


def build_comparison_table(
    wba_map: dict[int, list[Path]],
    sba_map: dict[int, list[Path]],
) -> pd.DataFrame:
    """Build the participant level empirical model comparison table."""
    common_ids = sorted(set(wba_map).intersection(sba_map))
    rows: list[dict] = []

    for participant_id in common_ids:
        wba_metrics = compute_model_metrics(wba_map[participant_id])
        sba_metrics = compute_model_metrics(sba_map[participant_id])

        waic_wba = wba_metrics["WAIC"]
        waic_sba = sba_metrics["WAIC"]
        total_wba = wba_metrics["total_log_lik"]
        total_sba = sba_metrics["total_log_lik"]

        if waic_wba < waic_sba:
            better_waic = "WBA"
        elif waic_sba < waic_wba:
            better_waic = "SBA"
        else:
            better_waic = "tie"

        if total_wba > total_sba:
            better_loglik = "WBA"
        elif total_sba > total_wba:
            better_loglik = "SBA"
        else:
            better_loglik = "tie"

        rows.append(
            {
                "participant_id": participant_id,
                "total_log_lik_wba": total_wba,
                "total_log_lik_sba": total_sba,
                "mean_log_lik_wba": wba_metrics["mean_log_lik"],
                "mean_log_lik_sba": sba_metrics["mean_log_lik"],
                "WAIC_wba": waic_wba,
                "WAIC_sba": waic_sba,
                "elpd_waic_wba": wba_metrics["elpd_waic"],
                "elpd_waic_sba": sba_metrics["elpd_waic"],
                "better_model_by_WAIC": better_waic,
                "better_model_by_total_log_lik": better_loglik,
            }
        )

    return pd.DataFrame(rows)


def save_waic_difference_plot(df: pd.DataFrame, output_path: Path) -> None:
    """Plot participant level WAIC difference (WAIC_SBA - WAIC_WBA)."""
    plot_df = df.copy()
    plot_df["waic_difference"] = plot_df["WAIC_sba"] - plot_df["WAIC_wba"]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = ["#b55d4c" if value > 0 else "#54728c" for value in plot_df["waic_difference"]]
    ax.bar(plot_df["participant_id"].astype(str), plot_df["waic_difference"], color=colors, edgecolor="black")
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_title("Participant-Level WAIC Difference (WAIC_SBA - WAIC_WBA)")
    ax.set_xlabel("Participant ID")
    ax.set_ylabel("WAIC difference")
    ax.tick_params(axis="x", labelrotation=90)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_waic_difference_histogram(df: pd.DataFrame, output_path: Path) -> None:
    """Plot the distribution of participant level WAIC differences."""
    plot_df = df.copy()
    plot_df["waic_difference"] = plot_df["WAIC_sba"] - plot_df["WAIC_wba"]

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.hist(plot_df["waic_difference"], bins=15, color="#8d6c4b", edgecolor="black", alpha=0.8)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_title("Distribution of WAIC Differences Across Participants")
    ax.set_xlabel("WAIC_SBA - WAIC_WBA")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def print_summary(df: pd.DataFrame) -> None:
    """Print a concise terminal summary of empirical model comparison."""
    n_participants = len(df)
    waic_diff = df["WAIC_sba"] - df["WAIC_wba"]
    loglik_diff = df["total_log_lik_wba"] - df["total_log_lik_sba"]

    n_wba_better_waic = int((df["better_model_by_WAIC"] == "WBA").sum())
    n_sba_better_waic = int((df["better_model_by_WAIC"] == "SBA").sum())

    print("=" * 90)
    print("Empirical all-participants model comparison")
    print("=" * 90)
    print(f"Participants successfully compared: {n_participants}")
    print(f"Participants where WBA has lower WAIC: {n_wba_better_waic}")
    print(f"Participants where SBA has lower WAIC: {n_sba_better_waic}")
    print(f"Mean WAIC difference (SBA - WBA): {waic_diff.mean():.3f}")
    print(f"Mean total_log_lik difference (WBA - SBA): {loglik_diff.mean():.3f}")


def main() -> None:
    """Reuse saved empirical fits to compare SBA and WBA across participants."""
    project_root = Path(__file__).resolve().parents[1]
    empirical_dir = project_root / "results" / "fits" / "empirical"
    figures_dir = project_root / "figures" / "empirical_model_comparison"
    ensure_directories(figures_dir)

    wba_per_dir = empirical_dir / "wba_per_participant"
    sba_per_dir = empirical_dir / "sba_per_participant"

    wba_map = build_participant_chain_map(wba_per_dir, "wba_model")
    sba_map = build_participant_chain_map(sba_per_dir, "sba_model")

    comparison_df = build_comparison_table(wba_map, sba_map)
    output_csv_path = empirical_dir / "empirical_all_participants_model_comparison.csv"
    comparison_df.to_csv(output_csv_path, index=False)

    waic_plot_path = figures_dir / "empirical_waic_difference_by_participant.png"
    waic_hist_path = figures_dir / "empirical_waic_difference_histogram.png"

    save_waic_difference_plot(comparison_df, waic_plot_path)
    save_waic_difference_histogram(comparison_df, waic_hist_path)

    print_summary(comparison_df)
    print("\nSaved comparison table to:")
    print(output_csv_path)
    print("\nSaved figures to:")
    print(waic_plot_path)
    print(waic_hist_path)


if __name__ == "__main__":
    main()
