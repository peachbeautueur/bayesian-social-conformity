"""Create participant-level posterior predictive checks for mean rating change.

This script reuses saved no-pooling empirical Stan outputs for all participants,
extracts posterior predictive draws (`y_rep`) for SBA and WBA, and summarizes
participant-level observed versus posterior predictive mean rating change.

Run from the project root:
    python scripts/17_ppc_participant_level_mean_change.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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


def extract_y_rep_matrix(draws_df: pd.DataFrame) -> np.ndarray:
    """Extract posterior predictive draws from concatenated CmdStan output."""
    y_rep_columns = [column for column in draws_df.columns if column.startswith("y_rep.")]
    if not y_rep_columns:
        raise ValueError("No y_rep columns were found in the saved Stan draws.")
    y_rep_columns = sorted(y_rep_columns, key=lambda name: int(name.split(".")[1]))
    return draws_df[y_rep_columns].to_numpy(dtype=float)


def round_and_clip_ratings(values: np.ndarray) -> np.ndarray:
    """Round predictive draws to the valid rating scale [1, 8]."""
    return np.clip(np.rint(values), 1, 8).astype(int)


def compute_model_predicted_mean_change(
    participant_df: pd.DataFrame,
    chain_paths: list[Path],
) -> float:
    """Compute posterior predictive mean rating change for one participant."""
    draws_df = load_cmdstan_chain_csvs(chain_paths)
    y_rep_matrix = extract_y_rep_matrix(draws_df)
    y_rep_discrete = round_and_clip_ratings(y_rep_matrix)
    first_rating = participant_df["FirstRating"].to_numpy(dtype=int)
    predictive_mean_change_draws = (y_rep_discrete - first_rating[np.newaxis, :]).mean(axis=1)
    return float(predictive_mean_change_draws.mean())


def build_summary_table(
    empirical_df: pd.DataFrame,
    wba_chain_map: dict[int, list[Path]],
    sba_chain_map: dict[int, list[Path]],
) -> pd.DataFrame:
    """Build the participant-level mean change PPC summary table."""
    common_ids = sorted(set(wba_chain_map).intersection(sba_chain_map))
    rows: list[dict[str, float | int]] = []

    for participant_id in common_ids:
        participant_df = empirical_df.loc[empirical_df["ID"] == participant_id].copy()
        observed_mean_change = float(
            (participant_df["SecondRating"].to_numpy(dtype=float) - participant_df["FirstRating"].to_numpy(dtype=float)).mean()
        )
        wba_predicted_mean_change = compute_model_predicted_mean_change(participant_df, wba_chain_map[participant_id])
        sba_predicted_mean_change = compute_model_predicted_mean_change(participant_df, sba_chain_map[participant_id])

        rows.append(
            {
                "participant_id": participant_id,
                "observed_mean_change": observed_mean_change,
                "wba_predicted_mean_change": wba_predicted_mean_change,
                "sba_predicted_mean_change": sba_predicted_mean_change,
                "wba_residual": wba_predicted_mean_change - observed_mean_change,
                "sba_residual": sba_predicted_mean_change - observed_mean_change,
            }
        )

    return pd.DataFrame(rows)


def save_observed_vs_predicted_plot(summary_df: pd.DataFrame, output_path: Path) -> None:
    """Save a scatter plot of observed vs predicted mean change."""
    all_values = np.concatenate(
        [
            summary_df["observed_mean_change"].to_numpy(dtype=float),
            summary_df["wba_predicted_mean_change"].to_numpy(dtype=float),
            summary_df["sba_predicted_mean_change"].to_numpy(dtype=float),
        ]
    )
    axis_min = float(all_values.min()) - 0.1
    axis_max = float(all_values.max()) + 0.1

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.scatter(
        summary_df["observed_mean_change"],
        summary_df["wba_predicted_mean_change"],
        color="#c65f46",
        alpha=0.8,
        edgecolor="black",
        label="WBA",
    )
    ax.scatter(
        summary_df["observed_mean_change"],
        summary_df["sba_predicted_mean_change"],
        color="#4f728c",
        alpha=0.8,
        edgecolor="black",
        marker="s",
        label="SBA",
    )
    ax.plot([axis_min, axis_max], [axis_min, axis_max], linestyle="--", color="black", linewidth=1.5)
    ax.set_xlim(axis_min, axis_max)
    ax.set_ylim(axis_min, axis_max)
    ax.set_title("Observed vs Predicted Participant Mean Rating Change")
    ax.set_xlabel("Observed mean rating change")
    ax.set_ylabel("Posterior predictive mean rating change")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_residual_histogram(summary_df: pd.DataFrame, output_path: Path) -> None:
    """Save a residual comparison histogram for SBA and WBA."""
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.hist(
        summary_df["wba_residual"],
        bins=12,
        alpha=0.65,
        color="#c65f46",
        edgecolor="black",
        label="WBA residual",
    )
    ax.hist(
        summary_df["sba_residual"],
        bins=12,
        alpha=0.45,
        color="#4f728c",
        edgecolor="black",
        label="SBA residual",
    )
    ax.axvline(0, color="black", linestyle="--", linewidth=1.5)
    ax.set_title("Participant-Level Residuals for Mean Rating Change")
    ax.set_xlabel("Predicted mean change - observed mean change")
    ax.set_ylabel("Count")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_side_by_side_point_plot(summary_df: pd.DataFrame, output_path: Path) -> None:
    """Save a side-by-side participant-level point plot."""
    plot_df = summary_df.sort_values("participant_id").reset_index(drop=True)
    x_positions = np.arange(len(plot_df))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(
        x_positions - 0.2,
        plot_df["observed_mean_change"],
        color="black",
        s=35,
        label="Observed",
    )
    ax.scatter(
        x_positions,
        plot_df["wba_predicted_mean_change"],
        color="#c65f46",
        s=35,
        label="WBA",
    )
    ax.scatter(
        x_positions + 0.2,
        plot_df["sba_predicted_mean_change"],
        color="#4f728c",
        marker="s",
        s=32,
        label="SBA",
    )
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_title("Participant-Level Mean Rating Change: Observed vs Predictive")
    ax.set_xlabel("Participant ID")
    ax.set_ylabel("Mean rating change")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(plot_df["participant_id"].astype(str), rotation=90)
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    """Create participant-level PPC summaries for mean rating change."""
    project_root = Path(__file__).resolve().parents[1]
    empirical_path = project_root / "data" / "Simonsen_clean.csv"
    empirical_dir = project_root / "results" / "fits" / "empirical"
    ppc_results_dir = project_root / "results" / "ppc"
    ppc_figures_dir = project_root / "figures" / "ppc"
    ensure_directories(ppc_results_dir, ppc_figures_dir)

    empirical_df = pd.read_csv(empirical_path)
    wba_chain_map = build_participant_chain_map(empirical_dir / "wba_per_participant", "wba_model")
    sba_chain_map = build_participant_chain_map(empirical_dir / "sba_per_participant", "sba_model")

    summary_df = build_summary_table(empirical_df, wba_chain_map, sba_chain_map)

    output_csv_path = ppc_results_dir / "participant_level_mean_change_ppc_summary.csv"
    summary_df.to_csv(output_csv_path, index=False)

    scatter_path = ppc_figures_dir / "ppc_participant_mean_change_observed_vs_predicted.png"
    residual_path = ppc_figures_dir / "ppc_participant_mean_change_residuals.png"
    point_plot_path = ppc_figures_dir / "ppc_participant_mean_change_point_plot.png"

    save_observed_vs_predicted_plot(summary_df, scatter_path)
    save_residual_histogram(summary_df, residual_path)
    save_side_by_side_point_plot(summary_df, point_plot_path)

    print("=" * 90)
    print("Participant-level PPC summary for mean rating change")
    print("=" * 90)
    print(f"Participants included: {len(summary_df)}")
    print(f"Mean absolute residual, WBA: {summary_df['wba_residual'].abs().mean():.3f}")
    print(f"Mean absolute residual, SBA: {summary_df['sba_residual'].abs().mean():.3f}")
    print("\nSaved summary table to:")
    print(output_csv_path)
    print("\nSaved figures to:")
    print(scatter_path)
    print(residual_path)
    print(point_plot_path)


if __name__ == "__main__":
    main()
