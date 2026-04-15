"""Create aggregate empirical posterior predictive checks for SBA and WBA.

This script reuses saved no-pooling empirical Stan outputs for all participants,
extracts posterior predictive draws (`y_rep`), aggregates them across the full
empirical dataset, and produces a small set of report-friendly posterior
predictive checks for SBA and WBA.

Run from the project root:
    python scripts/16_ppc_empirical_aggregate.py
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
    """Round predictive draws to the valid empirical rating scale [1, 8]."""
    return np.clip(np.rint(values), 1, 8).astype(int)


def compute_distribution_summary(
    observed_values: np.ndarray,
    replicated_values: np.ndarray,
    support_values: np.ndarray,
) -> pd.DataFrame:
    """Compute observed and posterior predictive proportions on a discrete support."""
    observed_counts = np.array([(observed_values == value).sum() for value in support_values], dtype=float)
    observed_proportions = observed_counts / observed_counts.sum()

    replicated_counts = np.stack(
        [(replicated_values == value).sum(axis=1) for value in support_values],
        axis=1,
    ).astype(float)
    replicated_proportions = replicated_counts / replicated_values.shape[1]

    return pd.DataFrame(
        {
            "value": support_values,
            "observed_proportion": observed_proportions,
            "predictive_mean_proportion": replicated_proportions.mean(axis=0),
            "predictive_lower_5": np.percentile(replicated_proportions, 5, axis=0),
            "predictive_upper_95": np.percentile(replicated_proportions, 95, axis=0),
        }
    )


def build_model_aggregate_replications(
    empirical_df: pd.DataFrame,
    participant_chain_map: dict[int, list[Path]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate discrete y_rep draws and observed trial variables across participants."""
    participant_ids = sorted(participant_chain_map)
    replicated_parts: list[np.ndarray] = []
    first_rating_parts: list[np.ndarray] = []
    observed_second_parts: list[np.ndarray] = []

    expected_draws: int | None = None
    for participant_id in participant_ids:
        participant_df = empirical_df.loc[empirical_df["ID"] == participant_id].copy()
        if participant_df.empty:
            raise ValueError(f"No empirical rows found for participant ID {participant_id}.")

        draws_df = load_cmdstan_chain_csvs(participant_chain_map[participant_id])
        y_rep_matrix = extract_y_rep_matrix(draws_df)
        y_rep_discrete = round_and_clip_ratings(y_rep_matrix)

        if expected_draws is None:
            expected_draws = y_rep_discrete.shape[0]
        elif y_rep_discrete.shape[0] != expected_draws:
            raise ValueError("Participants do not have the same number of posterior predictive draws.")

        replicated_parts.append(y_rep_discrete)
        first_rating_parts.append(participant_df["FirstRating"].to_numpy(dtype=int))
        observed_second_parts.append(participant_df["SecondRating"].to_numpy(dtype=int))

    replicated_second = np.concatenate(replicated_parts, axis=1)
    first_rating = np.concatenate(first_rating_parts).astype(int)
    observed_second = np.concatenate(observed_second_parts).astype(int)
    return replicated_second, first_rating, observed_second


def create_mean_summary_row(
    model_name: str,
    metric_name: str,
    observed_value: float,
    predictive_draws: np.ndarray,
) -> dict[str, float | str]:
    """Create one compact summary row for an aggregate mean PPC."""
    return {
        "model": model_name,
        "metric": metric_name,
        "observed_value": float(observed_value),
        "predictive_mean": float(predictive_draws.mean()),
        "predictive_sd": float(predictive_draws.std(ddof=1)),
        "predictive_lower_5": float(np.percentile(predictive_draws, 5)),
        "predictive_upper_95": float(np.percentile(predictive_draws, 95)),
    }


def save_distribution_plot(
    observed_summary: pd.DataFrame,
    wba_summary: pd.DataFrame,
    output_path: Path,
    title: str,
    x_label: str,
    include_sba: bool = False,
    sba_summary: pd.DataFrame | None = None,
) -> None:
    """Save an observed-vs-predictive discrete distribution comparison plot."""
    x_values = observed_summary["value"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(
        x_values - 0.18,
        observed_summary["observed_proportion"],
        width=0.32,
        color="#cccccc",
        edgecolor="black",
        label="Observed",
    )

    ax.errorbar(
        x_values,
        wba_summary["predictive_mean_proportion"],
        yerr=[
            wba_summary["predictive_mean_proportion"] - wba_summary["predictive_lower_5"],
            wba_summary["predictive_upper_95"] - wba_summary["predictive_mean_proportion"],
        ],
        fmt="o-",
        color="#c65f46",
        linewidth=1.5,
        markersize=5,
        capsize=3,
        label="WBA predictive",
    )

    if include_sba and sba_summary is not None:
        ax.errorbar(
            x_values + 0.18,
            sba_summary["predictive_mean_proportion"],
            yerr=[
                sba_summary["predictive_mean_proportion"] - sba_summary["predictive_lower_5"],
                sba_summary["predictive_upper_95"] - sba_summary["predictive_mean_proportion"],
            ],
            fmt="s--",
            color="#4f728c",
            linewidth=1.5,
            markersize=4.5,
            capsize=3,
            label="SBA predictive",
        )

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Proportion of trials")
    ax.set_xticks(x_values)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_mean_ppc_plot(
    observed_value: float,
    wba_draws: np.ndarray,
    output_path: Path,
    title: str,
    x_label: str,
    sba_draws: np.ndarray | None = None,
) -> None:
    """Save a posterior predictive check plot for an aggregate mean statistic."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.hist(wba_draws, bins=30, alpha=0.65, color="#c65f46", edgecolor="black", label="WBA predictive")

    if sba_draws is not None:
        ax.hist(sba_draws, bins=30, alpha=0.45, color="#4f728c", edgecolor="black", label="SBA predictive")

    ax.axvline(observed_value, color="black", linestyle="--", linewidth=2, label="Observed")
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Posterior predictive draw count")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    """Create aggregate empirical posterior predictive checks for SBA and WBA."""
    project_root = Path(__file__).resolve().parents[1]
    empirical_path = project_root / "data" / "Simonsen_clean.csv"
    empirical_dir = project_root / "results" / "fits" / "empirical"
    ppc_results_dir = project_root / "results" / "ppc"
    ppc_figures_dir = project_root / "figures" / "ppc"
    ensure_directories(ppc_results_dir, ppc_figures_dir)

    empirical_df = pd.read_csv(empirical_path)

    wba_chain_map = build_participant_chain_map(empirical_dir / "wba_per_participant", "wba_model")
    sba_chain_map = build_participant_chain_map(empirical_dir / "sba_per_participant", "sba_model")

    common_ids = sorted(set(wba_chain_map).intersection(sba_chain_map))
    empirical_df = empirical_df.loc[empirical_df["ID"].isin(common_ids)].copy()

    wba_replicated_second, first_rating, observed_second = build_model_aggregate_replications(empirical_df, wba_chain_map)
    sba_replicated_second, _, _ = build_model_aggregate_replications(empirical_df, sba_chain_map)

    observed_change = observed_second - first_rating
    wba_replicated_change = wba_replicated_second - first_rating[np.newaxis, :]
    sba_replicated_change = sba_replicated_second - first_rating[np.newaxis, :]

    rating_support = np.arange(1, 9)
    change_support = np.arange(-7, 8)

    observed_second_summary = compute_distribution_summary(observed_second, wba_replicated_second, rating_support)
    wba_second_summary = compute_distribution_summary(observed_second, wba_replicated_second, rating_support)
    sba_second_summary = compute_distribution_summary(observed_second, sba_replicated_second, rating_support)

    observed_change_summary = compute_distribution_summary(observed_change, wba_replicated_change, change_support)
    wba_change_summary = compute_distribution_summary(observed_change, wba_replicated_change, change_support)
    sba_change_summary = compute_distribution_summary(observed_change, sba_replicated_change, change_support)

    second_distribution_path = ppc_results_dir / "empirical_aggregate_second_rating_distribution.csv"
    change_distribution_path = ppc_results_dir / "empirical_aggregate_rating_change_distribution.csv"

    second_distribution_df = observed_second_summary.rename(
        columns={
            "predictive_mean_proportion": "wba_predictive_mean_proportion",
            "predictive_lower_5": "wba_predictive_lower_5",
            "predictive_upper_95": "wba_predictive_upper_95",
        }
    ).merge(
        sba_second_summary[["value", "predictive_mean_proportion", "predictive_lower_5", "predictive_upper_95"]].rename(
            columns={
                "predictive_mean_proportion": "sba_predictive_mean_proportion",
                "predictive_lower_5": "sba_predictive_lower_5",
                "predictive_upper_95": "sba_predictive_upper_95",
            }
        ),
        on="value",
        how="left",
    )
    second_distribution_df.to_csv(second_distribution_path, index=False)

    change_distribution_df = observed_change_summary.rename(
        columns={
            "predictive_mean_proportion": "wba_predictive_mean_proportion",
            "predictive_lower_5": "wba_predictive_lower_5",
            "predictive_upper_95": "wba_predictive_upper_95",
        }
    ).merge(
        sba_change_summary[["value", "predictive_mean_proportion", "predictive_lower_5", "predictive_upper_95"]].rename(
            columns={
                "predictive_mean_proportion": "sba_predictive_mean_proportion",
                "predictive_lower_5": "sba_predictive_lower_5",
                "predictive_upper_95": "sba_predictive_upper_95",
            }
        ),
        on="value",
        how="left",
    )
    change_distribution_df.to_csv(change_distribution_path, index=False)

    observed_mean_second = float(observed_second.mean())
    observed_mean_change = float(observed_change.mean())
    wba_mean_second_draws = wba_replicated_second.mean(axis=1)
    sba_mean_second_draws = sba_replicated_second.mean(axis=1)
    wba_mean_change_draws = wba_replicated_change.mean(axis=1)
    sba_mean_change_draws = sba_replicated_change.mean(axis=1)

    mean_summary_rows = [
        create_mean_summary_row("WBA", "mean_second_rating", observed_mean_second, wba_mean_second_draws),
        create_mean_summary_row("SBA", "mean_second_rating", observed_mean_second, sba_mean_second_draws),
        create_mean_summary_row("WBA", "mean_rating_change", observed_mean_change, wba_mean_change_draws),
        create_mean_summary_row("SBA", "mean_rating_change", observed_mean_change, sba_mean_change_draws),
    ]
    mean_summary_df = pd.DataFrame(mean_summary_rows)
    mean_summary_path = ppc_results_dir / "empirical_aggregate_mean_ppc_summary.csv"
    mean_summary_df.to_csv(mean_summary_path, index=False)

    save_distribution_plot(
        observed_summary=observed_second_summary,
        wba_summary=wba_second_summary,
        sba_summary=sba_second_summary,
        include_sba=True,
        output_path=ppc_figures_dir / "ppc_aggregate_second_rating_distribution.png",
        title="Aggregate Empirical PPC: SecondRating Distribution",
        x_label="SecondRating",
    )
    save_distribution_plot(
        observed_summary=observed_change_summary,
        wba_summary=wba_change_summary,
        sba_summary=sba_change_summary,
        include_sba=True,
        output_path=ppc_figures_dir / "ppc_aggregate_rating_change_distribution.png",
        title="Aggregate Empirical PPC: Rating Change Distribution",
        x_label="SecondRating - FirstRating",
    )
    save_mean_ppc_plot(
        observed_value=observed_mean_second,
        wba_draws=wba_mean_second_draws,
        sba_draws=sba_mean_second_draws,
        output_path=ppc_figures_dir / "ppc_aggregate_mean_second_rating.png",
        title="Aggregate Empirical PPC: Mean SecondRating",
        x_label="Posterior predictive mean SecondRating",
    )
    save_mean_ppc_plot(
        observed_value=observed_mean_change,
        wba_draws=wba_mean_change_draws,
        sba_draws=sba_mean_change_draws,
        output_path=ppc_figures_dir / "ppc_aggregate_mean_rating_change.png",
        title="Aggregate Empirical PPC: Mean Rating Change",
        x_label="Posterior predictive mean rating change",
    )

    print("=" * 90)
    print("Aggregate empirical posterior predictive checks")
    print("=" * 90)
    print(f"Participants included: {len(common_ids)}")
    print(f"Observed mean SecondRating: {observed_mean_second:.3f}")
    print(f"WBA predictive mean SecondRating: {wba_mean_second_draws.mean():.3f}")
    print(f"SBA predictive mean SecondRating: {sba_mean_second_draws.mean():.3f}")
    print(f"Observed mean rating change: {observed_mean_change:.3f}")
    print(f"WBA predictive mean rating change: {wba_mean_change_draws.mean():.3f}")
    print(f"SBA predictive mean rating change: {sba_mean_change_draws.mean():.3f}")
    print("\nSaved summary CSV files to:")
    print(second_distribution_path)
    print(change_distribution_path)
    print(mean_summary_path)
    print("\nSaved figures to:")
    print(ppc_figures_dir / "ppc_aggregate_second_rating_distribution.png")
    print(ppc_figures_dir / "ppc_aggregate_rating_change_distribution.png")
    print(ppc_figures_dir / "ppc_aggregate_mean_second_rating.png")
    print(ppc_figures_dir / "ppc_aggregate_mean_rating_change.png")


if __name__ == "__main__":
    main()
