"""Visualize simple prior-posterior updates for empirical WBA fits.

This script reads the saved participant-level WBA empirical summary table,
recreates the weakly informative priors used in `stan/wba_model.stan`, and
compares those priors with the distribution of participant-level posterior mean
estimates. It is intended as a transparent, report-friendly summary rather than
as a full posterior analysis.

Run from the project root:
    python scripts/18_prior_posterior_updates_wba_empirical.py
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


def truncated_normal_density(x: np.ndarray, mean: float, sd: float, lower: float, upper: float) -> np.ndarray:
    """Return a numerically normalized truncated normal density on a fixed grid.

    This avoids any SciPy dependency by normalizing the unnormalized normal
    density numerically over the bounded support used in the Stan model.
    """
    density = np.zeros_like(x, dtype=float)
    in_support = (x >= lower) & (x <= upper)
    z = (x[in_support] - mean) / sd
    density[in_support] = np.exp(-0.5 * z**2)
    area = np.trapz(density, x)
    if area <= 0:
        raise ValueError("Prior density normalization failed.")
    return density / area


def build_summary_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Create a compact prior-posterior summary table."""
    parameter_specs = [
        {
            "parameter": "w_direct",
            "column": "posterior_mean_w_direct",
            "prior_description": "truncated normal(1, 1), bounds [0, 5]",
        },
        {
            "parameter": "w_social",
            "column": "posterior_mean_w_social",
            "prior_description": "truncated normal(1, 1), bounds [0, 5]",
        },
        {
            "parameter": "sigma",
            "column": "posterior_mean_sigma",
            "prior_description": "truncated normal(0.5, 0.5), bounds [1e-6, 2]",
        },
    ]

    rows: list[dict[str, float | str]] = []
    for spec in parameter_specs:
        values = summary_df[spec["column"]].to_numpy(dtype=float)
        rows.append(
            {
                "parameter": spec["parameter"],
                "prior_center_or_description": spec["prior_description"],
                "posterior_mean_of_means": float(values.mean()),
                "posterior_sd_of_means": float(values.std(ddof=1)),
                "posterior_min": float(values.min()),
                "posterior_max": float(values.max()),
            }
        )

    return pd.DataFrame(rows)


def save_prior_posterior_figure(summary_df: pd.DataFrame, output_path: Path) -> None:
    """Save a three-panel prior vs posterior-mean comparison figure."""
    parameter_specs = [
        {
            "title": "w_direct",
            "column": "posterior_mean_w_direct",
            "prior_mean": 1.0,
            "prior_sd": 1.0,
            "lower": 0.0,
            "upper": 5.0,
            "color": "#c65f46",
        },
        {
            "title": "w_social",
            "column": "posterior_mean_w_social",
            "prior_mean": 1.0,
            "prior_sd": 1.0,
            "lower": 0.0,
            "upper": 5.0,
            "color": "#4f728c",
        },
        {
            "title": "sigma",
            "column": "posterior_mean_sigma",
            "prior_mean": 0.5,
            "prior_sd": 0.5,
            "lower": 1e-6,
            "upper": 2.0,
            "color": "#8a6d3b",
        },
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))

    for ax, spec in zip(axes, parameter_specs):
        values = summary_df[spec["column"]].to_numpy(dtype=float)
        x_grid = np.linspace(spec["lower"], spec["upper"], 500)
        prior_density = truncated_normal_density(
            x=x_grid,
            mean=spec["prior_mean"],
            sd=spec["prior_sd"],
            lower=spec["lower"],
            upper=spec["upper"],
        )

        ax.hist(
            values,
            bins=12,
            density=True,
            alpha=0.7,
            color=spec["color"],
            edgecolor="black",
            label="Participant posterior means",
        )
        ax.plot(
            x_grid,
            prior_density,
            color="black",
            linewidth=2,
            label="Prior density",
        )
        ax.set_title(spec["title"])
        ax.set_xlabel("Parameter value")
        ax.set_ylabel("Density")
        ax.set_xlim(spec["lower"], spec["upper"])
        ax.legend(frameon=False, fontsize=8)

    fig.suptitle("WBA Empirical Prior-Posterior Update Summary", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Create prior-posterior update summaries for empirical WBA fits."""
    project_root = Path(__file__).resolve().parents[1]
    summary_path = project_root / "results" / "fits" / "empirical" / "wba_empirical_all_participants_summary.csv"
    results_dir = project_root / "results" / "prior_posterior"
    figures_dir = project_root / "figures" / "prior_posterior"
    ensure_directories(results_dir, figures_dir)

    summary_df = pd.read_csv(summary_path)
    success_df = summary_df.loc[summary_df["status"] == "success"].copy()

    output_summary_path = results_dir / "wba_empirical_prior_posterior_summary.csv"
    output_figure_path = figures_dir / "wba_empirical_prior_posterior_updates.png"

    compact_summary_df = build_summary_table(success_df)
    compact_summary_df.to_csv(output_summary_path, index=False)
    save_prior_posterior_figure(success_df, output_figure_path)

    print("=" * 90)
    print("WBA empirical prior-posterior update summary")
    print("=" * 90)
    print(f"Participants included: {len(success_df)}")
    print("\nSaved summary table to:")
    print(output_summary_path)
    print("\nSaved figure to:")
    print(output_figure_path)


if __name__ == "__main__":
    main()
