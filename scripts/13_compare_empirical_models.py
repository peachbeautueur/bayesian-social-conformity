"""Compare empirical no pooling SBA and WBA fit summaries across participants.

This script reads the saved participant level fit summaries.
It merges them into one comparison table.
It prints a short terminal summary.
It makes a few simple figures.

Run from the project root:
    python scripts/13_compare_empirical_models.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def ensure_directories(*paths: Path) -> None:
    """Create output directories if they do not already exist."""
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def load_summary(csv_path: Path) -> pd.DataFrame:
    """Load a participant level empirical fit summary."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Summary file not found: {csv_path}")
    return pd.read_csv(csv_path)


def build_comparison_table(wba_df: pd.DataFrame, sba_df: pd.DataFrame) -> pd.DataFrame:
    """Merge the WBA and SBA summaries into one comparison table."""
    wba_subset = wba_df[
        [
            "participant_id",
            "n_trials",
            "posterior_mean_sigma",
            "posterior_mean_w_direct",
            "posterior_mean_w_social",
            "divergences",
            "R_hat_max",
            "status",
        ]
    ].rename(
        columns={
            "posterior_mean_sigma": "posterior_mean_sigma_wba",
            "posterior_mean_w_direct": "posterior_mean_w_direct",
            "posterior_mean_w_social": "posterior_mean_w_social",
            "divergences": "divergences_wba",
            "R_hat_max": "R_hat_max_wba",
            "status": "status_wba",
        }
    )

    sba_subset = sba_df[
        [
            "participant_id",
            "n_trials",
            "posterior_mean_sigma",
            "divergences",
            "R_hat_max",
            "status",
        ]
    ].rename(
        columns={
            "n_trials": "n_trials_sba",
            "posterior_mean_sigma": "posterior_mean_sigma_sba",
            "divergences": "divergences_sba",
            "R_hat_max": "R_hat_max_sba",
            "status": "status_sba",
        }
    )

    comparison_df = wba_subset.merge(sba_subset, on="participant_id", how="outer")
    comparison_df["n_trials"] = comparison_df["n_trials"].fillna(comparison_df["n_trials_sba"])
    comparison_df["sigma_difference"] = (
        comparison_df["posterior_mean_sigma_sba"] - comparison_df["posterior_mean_sigma_wba"]
    )

    ordered_columns = [
        "participant_id",
        "n_trials",
        "posterior_mean_sigma_wba",
        "posterior_mean_sigma_sba",
        "sigma_difference",
        "posterior_mean_w_direct",
        "posterior_mean_w_social",
        "divergences_wba",
        "divergences_sba",
        "R_hat_max_wba",
        "R_hat_max_sba",
        "status_wba",
        "status_sba",
    ]
    return comparison_df[ordered_columns].sort_values("participant_id").reset_index(drop=True)


def save_sigma_comparison_plot(df: pd.DataFrame, output_path: Path) -> None:
    """Plot posterior mean sigma for SBA and WBA as separate points."""
    plot_df = df.copy()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x_positions = range(len(plot_df))
    wba_x = [x - 0.12 for x in x_positions]
    sba_x = [x + 0.12 for x in x_positions]

    ax.scatter(
        wba_x,
        plot_df["posterior_mean_sigma_wba"],
        marker="o",
        s=45,
        label="WBA sigma",
        color="#54728c",
        alpha=0.9,
    )
    ax.scatter(
        sba_x,
        plot_df["posterior_mean_sigma_sba"],
        marker="o",
        s=45,
        label="SBA sigma",
        color="#b55d4c",
        alpha=0.9,
    )

    ax.set_title("Posterior Mean Sigma Across Participants")
    ax.set_xlabel("Participant ID")
    ax.set_ylabel("Posterior mean sigma")
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(plot_df["participant_id"].astype(str), rotation=90)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_sigma_difference_plot(df: pd.DataFrame, output_path: Path) -> None:
    """Plot the participant level sigma difference between SBA and WBA."""
    plot_df = df.copy()
    fig, ax = plt.subplots(figsize=(10, 5.5))

    colors = ["#b55d4c" if value > 0 else "#54728c" for value in plot_df["sigma_difference"]]
    ax.bar(plot_df["participant_id"].astype(str), plot_df["sigma_difference"], color=colors, edgecolor="black")
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_title("Sigma Difference Across Participants (sigma_SBA - sigma_WBA)")
    ax.set_xlabel("Participant ID")
    ax.set_ylabel("sigma_SBA - sigma_WBA")
    ax.tick_params(axis="x", labelrotation=60)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_wba_parameter_distribution_plot(df: pd.DataFrame, output_path: Path) -> None:
    """Plot separate distributions for WBA parameter estimates."""
    plot_df = df.copy()
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

    axes[0].hist(
        plot_df["posterior_mean_w_direct"].dropna(),
        bins=15,
        alpha=0.75,
        color="#5c8b68",
        edgecolor="black",
    )
    axes[1].hist(
        plot_df["posterior_mean_w_social"].dropna(),
        bins=15,
        alpha=0.75,
        color="#b55d4c",
        edgecolor="black",
    )

    axes[0].set_title("WBA: posterior_mean_w_direct")
    axes[0].set_xlabel("Posterior mean")
    axes[0].set_ylabel("Count")

    axes[1].set_title("WBA: posterior_mean_w_social")
    axes[1].set_xlabel("Posterior mean")

    fig.suptitle("Distribution of WBA Parameter Estimates Across Participants")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def print_terminal_summary(df: pd.DataFrame) -> None:
    """Print a concise textual summary of the empirical model comparison."""
    successful_df = df.loc[(df["status_wba"] == "success") & (df["status_sba"] == "success")].copy()

    n_success = len(successful_df)
    mean_sigma_sba = successful_df["posterior_mean_sigma_sba"].mean()
    mean_sigma_wba = successful_df["posterior_mean_sigma_wba"].mean()
    mean_sigma_difference = successful_df["sigma_difference"].mean()
    n_sigma_sba_greater = int((successful_df["posterior_mean_sigma_sba"] > successful_df["posterior_mean_sigma_wba"]).sum())

    print("=" * 90)
    print("Empirical model comparison summary")
    print("=" * 90)
    print(f"Participants successfully fitted in both models: {n_success}")
    print(f"Mean sigma for SBA: {mean_sigma_sba:.3f}")
    print(f"Mean sigma for WBA: {mean_sigma_wba:.3f}")
    print(f"Mean sigma difference (SBA - WBA): {mean_sigma_difference:.3f}")
    print(f"Participants with sigma_SBA > sigma_WBA: {n_sigma_sba_greater}")


def main() -> None:
    """Merge saved empirical fit summaries and create comparison outputs."""
    project_root = Path(__file__).resolve().parents[1]
    empirical_dir = project_root / "results" / "fits" / "empirical"
    figures_dir = project_root / "figures" / "empirical"
    ensure_directories(figures_dir)

    wba_path = empirical_dir / "wba_empirical_all_participants_summary.csv"
    sba_path = empirical_dir / "sba_empirical_all_participants_summary.csv"

    wba_df = load_summary(wba_path)
    sba_df = load_summary(sba_path)
    comparison_df = build_comparison_table(wba_df, sba_df)

    output_csv_path = empirical_dir / "empirical_model_comparison_summary.csv"
    comparison_df.to_csv(output_csv_path, index=False)

    sigma_plot_path = figures_dir / "empirical_sigma_comparison.png"
    sigma_difference_path = figures_dir / "empirical_sigma_difference.png"
    wba_parameter_path = figures_dir / "empirical_wba_parameter_distribution.png"

    save_sigma_comparison_plot(comparison_df, sigma_plot_path)
    save_sigma_difference_plot(comparison_df, sigma_difference_path)
    save_wba_parameter_distribution_plot(comparison_df, wba_parameter_path)

    print_terminal_summary(comparison_df)
    print("\nSaved comparison table to:")
    print(output_csv_path)
    print("\nSaved figures to:")
    print(sigma_plot_path)
    print(sigma_difference_path)
    print(wba_parameter_path)


if __name__ == "__main__":
    main()
