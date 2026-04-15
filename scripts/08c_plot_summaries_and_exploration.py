"""Create extra empirical exploration figures.

This script uses the Simonsen dataset to generate a small set of figures.
Synthetic recovery figures are handled in other scripts.
That keeps this script focused on empirical exploration.

Run from the project root:
    python scripts/08c_plot_summaries_and_exploration.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def ensure_directories(*paths: Path) -> None:
    """Create output directories if they do not already exist."""
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def load_empirical_data(csv_path: Path) -> pd.DataFrame:
    """Load the empirical Simonsen dataset."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Empirical dataset not found: {csv_path}")
    return pd.read_csv(csv_path)


def prepare_empirical_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived variables used in the empirical exploration figures."""
    result = df.copy()
    result["social_gap"] = result["GroupRating"] - result["FirstRating"]
    result["rating_change"] = result["SecondRating"] - result["FirstRating"]
    return result


def save_social_gap_change_plot(df: pd.DataFrame, output_path: Path) -> None:
    """Plot rating change against social gap with a binned trend overlay."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df["social_gap"], df["rating_change"], alpha=0.2, s=18, color="#54728c")

    binned = (
        df.groupby("social_gap", as_index=False)["rating_change"]
        .mean()
        .sort_values("social_gap")
    )
    ax.plot(
        binned["social_gap"],
        binned["rating_change"],
        color="#b55d4c",
        marker="o",
        linewidth=2,
        label="Mean by social gap",
    )

    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.axvline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_title("Rating Change as a Function of Social Gap")
    ax.set_xlabel("GroupRating - FirstRating")
    ax.set_ylabel("SecondRating - FirstRating")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_rating_change_histogram(df: pd.DataFrame, output_path: Path) -> None:
    """Plot the distribution of rating changes."""
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = range(int(df["rating_change"].min()) - 1, int(df["rating_change"].max()) + 2)
    ax.hist(df["rating_change"], bins=bins, edgecolor="black", color="#8a6d4b")
    ax.set_title("Distribution of Rating Change")
    ax.set_xlabel("SecondRating - FirstRating")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_rating_distribution_comparison(df: pd.DataFrame, output_path: Path) -> None:
    """Compare the distributions of first, group, and second ratings."""
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    bins = [x - 0.5 for x in range(1, 10)]

    ax.hist(df["FirstRating"], bins=bins, alpha=0.5, label="FirstRating", color="#54728c", edgecolor="black")
    ax.hist(df["GroupRating"], bins=bins, alpha=0.5, label="GroupRating", color="#b55d4c", edgecolor="black")
    ax.hist(df["SecondRating"], bins=bins, alpha=0.5, label="SecondRating", color="#6b8f5e", edgecolor="black")

    ax.set_xticks(range(1, 9))
    ax.set_title("Rating Distributions in Simonsen Data")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    """Create clear empirical exploration figures."""
    project_root = Path(__file__).resolve().parents[1]
    exploration_report_dir = project_root / "figures" / "exploration_report"
    simonsen_path = project_root / "data" / "Simonsen_clean.csv"

    ensure_directories(exploration_report_dir)

    simonsen_df = prepare_empirical_derived_columns(load_empirical_data(simonsen_path))
    save_social_gap_change_plot(
        simonsen_df,
        output_path=exploration_report_dir / "simonsen_social_gap_vs_rating_change.png",
    )
    save_rating_change_histogram(
        simonsen_df,
        output_path=exploration_report_dir / "simonsen_rating_change_histogram.png",
    )
    save_rating_distribution_comparison(
        simonsen_df,
        output_path=exploration_report_dir / "simonsen_rating_distribution_comparison.png",
    )

    print("\nSaved exploration figures to:")
    print(exploration_report_dir / "simonsen_social_gap_vs_rating_change.png")
    print(exploration_report_dir / "simonsen_rating_change_histogram.png")
    print(exploration_report_dir / "simonsen_rating_distribution_comparison.png")


if __name__ == "__main__":
    main()
