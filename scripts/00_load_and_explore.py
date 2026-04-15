"""Load and explore all CSV datasets for the Bayesian social conformity project.

This script loads every CSV file from `data/`.
It prints basic information for each dataset.
It adds rating change and social gap.
It saves summary tables.
It saves basic plots.

Run from the project root:
    python scripts/00_load_and_explore.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


EXPECTED_COLUMNS = [
    "ID",
    "FaceID",
    "FirstRating",
    "GroupRating",
    "SecondRating",
]

RATING_COLUMNS = ["FirstRating", "GroupRating", "SecondRating"]


def ensure_directories(*paths: Path) -> None:
    """Create output directories if they do not already exist."""
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def print_section(title: str) -> None:
    """Print a small console section header."""
    print(f"\n{'=' * 80}")
    print(title)
    print(f"{'=' * 80}")


def check_expected_columns(df: pd.DataFrame, csv_path: Path) -> None:
    """Raise an error if required columns are missing."""
    missing_columns = [column for column in EXPECTED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Dataset '{csv_path.name}' is missing required columns: {missing_columns}"
        )


def build_summary_table(
    df: pd.DataFrame,
    filename: str,
    rating_range_summary: dict[str, int],
    participant_trials: pd.Series,
) -> pd.DataFrame:
    """Assemble a one row summary table for a dataset."""
    summary = {
        "filename": filename,
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "n_missing_total": int(df.isna().sum().sum()),
        "n_unique_participants": int(df["ID"].nunique(dropna=True)),
        "n_unique_faces": int(df["FaceID"].nunique(dropna=True)),
        "participant_trials_mean": float(participant_trials.mean()),
        "participant_trials_min": int(participant_trials.min()),
        "participant_trials_max": int(participant_trials.max()),
    }
    summary.update(rating_range_summary)
    return pd.DataFrame([summary])


def summarize_descriptives(df: pd.DataFrame) -> pd.DataFrame:
    """Create descriptive statistics for rating variables."""
    columns_to_summarize = RATING_COLUMNS + ["rating_change", "social_gap"]
    return df[columns_to_summarize].describe().transpose()


def save_plot_histogram(series: pd.Series, title: str, xlabel: str, output_path: Path) -> None:
    """Save a histogram for a single variable."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(series.dropna(), bins=20, edgecolor="black")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_scatter_plot(df: pd.DataFrame, output_path: Path) -> None:
    """Save a scatter plot of social gap versus rating change."""
    fig, ax = plt.subplots(figsize=(7, 6))
    clean_df = df[["social_gap", "rating_change"]].dropna()
    ax.scatter(clean_df["social_gap"], clean_df["rating_change"], alpha=0.5)
    ax.set_title("Social Gap vs Rating Change")
    ax.set_xlabel("social_gap")
    ax.set_ylabel("rating_change")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def explore_dataset(csv_path: Path, results_dir: Path, figures_dir: Path) -> None:
    """Load, inspect, summarize, and plot a single dataset."""
    dataset_name = csv_path.stem
    print_section(f"Exploring dataset: {csv_path.name}")

    df = pd.read_csv(csv_path)
    check_expected_columns(df, csv_path)

    # Derived variables used later for exploratory analysis and modeling.
    df["rating_change"] = df["SecondRating"] - df["FirstRating"]
    df["social_gap"] = df["GroupRating"] - df["FirstRating"]

    print(f"Filename: {csv_path.name}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    missing_values = df.isna().sum()
    print("\nMissing values per column:")
    print(missing_values.to_string())

    n_participants = df["ID"].nunique(dropna=True)
    n_faces = df["FaceID"].nunique(dropna=True)
    print(f"\nUnique participants (ID): {n_participants}")
    print(f"Unique faces (FaceID): {n_faces}")

    participant_trials = df.groupby("ID").size()
    print("\nTrial counts per participant:")
    print(f"Mean: {participant_trials.mean():.2f}")
    print(f"Min: {participant_trials.min()}")
    print(f"Max: {participant_trials.max()}")

    rating_range_summary: dict[str, int] = {}
    print("\nValues outside the expected 1-8 range:")
    for column in RATING_COLUMNS:
        valid_mask = df[column].between(1, 8, inclusive="both")
        outside_range_count = int((~valid_mask & df[column].notna()).sum())
        rating_range_summary[f"{column}_outside_1_8"] = outside_range_count
        print(f"{column}: {outside_range_count}")

    dataset_results_dir = results_dir
    dataset_figures_dir = figures_dir / dataset_name
    ensure_directories(dataset_results_dir, dataset_figures_dir)

    summary_table = build_summary_table(
        df=df,
        filename=csv_path.name,
        rating_range_summary=rating_range_summary,
        participant_trials=participant_trials,
    )
    descriptives_table = summarize_descriptives(df)

    summary_output_path = dataset_results_dir / f"{dataset_name}_summary.csv"
    descriptives_output_path = dataset_results_dir / f"{dataset_name}_descriptives.csv"
    missing_output_path = dataset_results_dir / f"{dataset_name}_missing_values.csv"

    summary_table.to_csv(summary_output_path, index=False)
    descriptives_table.to_csv(descriptives_output_path)
    missing_values.rename("missing_count").to_csv(missing_output_path, header=True)

    save_plot_histogram(
        df["FirstRating"],
        title=f"{dataset_name}: FirstRating",
        xlabel="FirstRating",
        output_path=dataset_figures_dir / "hist_first_rating.png",
    )
    save_plot_histogram(
        df["GroupRating"],
        title=f"{dataset_name}: GroupRating",
        xlabel="GroupRating",
        output_path=dataset_figures_dir / "hist_group_rating.png",
    )
    save_plot_histogram(
        df["SecondRating"],
        title=f"{dataset_name}: SecondRating",
        xlabel="SecondRating",
        output_path=dataset_figures_dir / "hist_second_rating.png",
    )
    save_plot_histogram(
        df["rating_change"],
        title=f"{dataset_name}: rating_change",
        xlabel="rating_change",
        output_path=dataset_figures_dir / "hist_rating_change.png",
    )
    save_plot_histogram(
        df["social_gap"],
        title=f"{dataset_name}: social_gap",
        xlabel="social_gap",
        output_path=dataset_figures_dir / "hist_social_gap.png",
    )
    save_scatter_plot(
        df,
        output_path=dataset_figures_dir / "scatter_social_gap_vs_rating_change.png",
    )

    print(f"\nSaved summary files to: {dataset_results_dir}")
    print(f"Saved plots to: {dataset_figures_dir}")


def main() -> None:
    """Run exploratory analysis for every CSV file in the data directory."""
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    results_dir = project_root / "results" / "exploration"
    figures_dir = project_root / "figures" / "exploration"

    ensure_directories(results_dir, figures_dir)

    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    print_section("Bayesian Social Conformity: Data Exploration")
    print(f"Project root: {project_root}")
    print(f"Found {len(csv_files)} CSV file(s) in: {data_dir}")

    for csv_path in csv_files:
        explore_dataset(csv_path, results_dir=results_dir, figures_dir=figures_dir)

    print_section("Exploration complete")
    print(f"Summary outputs: {results_dir}")
    print(f"Figure outputs: {figures_dir}")


if __name__ == "__main__":
    main()
