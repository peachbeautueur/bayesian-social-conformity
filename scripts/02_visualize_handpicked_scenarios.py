"""Visualize SBA and WBA predictions for five selected scenarios.

This script keeps the beta update logic from the prototype agents.
It applies that logic to a small set of trust rating scenarios.
The goal is to show how SBA and different WBA settings behave.

Run from the project root:
    python scripts/02_visualize_handpicked_scenarios.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ALPHA0 = 1.0
BETA0 = 1.0
N_DIRECT = 7
N_SOCIAL = 7
RATING_MIN = 1
RATING_MAX = 8


def ensure_directories(*paths: Path) -> None:
    """Create output directories if they do not already exist."""
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def rating_to_probability(rating: pd.Series | np.ndarray | float) -> pd.Series | np.ndarray | float:
    """Map a 1-8 rating onto a probability in [0, 1]."""
    return (rating - RATING_MIN) / (RATING_MAX - RATING_MIN)


def probability_to_rating(
    probability: pd.Series | np.ndarray | float,
) -> pd.Series | np.ndarray | float:
    """Map a probability in [0, 1] back onto the 1-8 rating scale."""
    return RATING_MIN + (RATING_MAX - RATING_MIN) * probability


def sba_predict(
    df: pd.DataFrame,
    alpha0: float = ALPHA0,
    beta0: float = BETA0,
    n_direct: float = N_DIRECT,
    n_social: float = N_SOCIAL,
) -> pd.DataFrame:
    """Predict updated ratings under the Simple Bayesian Agent."""
    result = df.copy()

    p_direct = rating_to_probability(result["FirstRating"])
    p_social = rating_to_probability(result["GroupRating"])

    k_direct = p_direct * n_direct
    k_social = p_social * n_social

    alpha_post = alpha0 + k_direct + k_social
    beta_post = beta0 + (n_direct - k_direct) + (n_social - k_social)
    posterior_mean = alpha_post / (alpha_post + beta_post)

    predicted_continuous = probability_to_rating(posterior_mean)
    predicted_rounded = np.clip(np.rint(predicted_continuous), RATING_MIN, RATING_MAX).astype(int)

    result["predicted_second_rating_continuous"] = predicted_continuous
    result["predicted_second_rating_rounded"] = predicted_rounded
    result["predicted_change"] = predicted_continuous - result["FirstRating"]

    return result


def wba_predict(
    df: pd.DataFrame,
    w_direct: float,
    w_social: float,
    alpha0: float = ALPHA0,
    beta0: float = BETA0,
    n_direct: float = N_DIRECT,
    n_social: float = N_SOCIAL,
) -> pd.DataFrame:
    """Predict updated ratings under the Weighted Bayesian Agent."""
    result = df.copy()

    p_direct = rating_to_probability(result["FirstRating"])
    p_social = rating_to_probability(result["GroupRating"])

    k_direct = p_direct * n_direct
    k_social = p_social * n_social

    alpha_post = alpha0 + w_direct * k_direct + w_social * k_social
    beta_post = (
        beta0
        + w_direct * (n_direct - k_direct)
        + w_social * (n_social - k_social)
    )
    posterior_mean = alpha_post / (alpha_post + beta_post)

    predicted_continuous = probability_to_rating(posterior_mean)
    predicted_rounded = np.clip(np.rint(predicted_continuous), RATING_MIN, RATING_MAX).astype(int)

    result["predicted_second_rating_continuous"] = predicted_continuous
    result["predicted_second_rating_rounded"] = predicted_rounded
    result["predicted_change"] = predicted_continuous - result["FirstRating"]

    return result


def build_handpicked_scenarios() -> pd.DataFrame:
    """Construct the five requested scenarios."""
    scenarios = pd.DataFrame(
        {
            "scenario": [
                "agreement",
                "small positive disagreement",
                "small negative disagreement",
                "large positive disagreement",
                "large negative disagreement",
            ],
            "FirstRating": [4, 4, 5, 2, 7],
            "GroupRating": [4, 5, 4, 7, 2],
        }
    )
    scenarios["social_gap"] = scenarios["GroupRating"] - scenarios["FirstRating"]
    return scenarios


def build_scenario_table() -> pd.DataFrame:
    """Compute SBA and WBA predictions for the five selected scenarios."""
    scenarios = build_handpicked_scenarios()

    scenario_table = scenarios.copy()

    sba_results = sba_predict(scenarios)
    scenario_table["SBA_predicted_second_rating_continuous"] = (
        sba_results["predicted_second_rating_continuous"]
    )
    scenario_table["SBA_predicted_second_rating_rounded"] = (
        sba_results["predicted_second_rating_rounded"]
    )
    scenario_table["SBA_predicted_change"] = sba_results["predicted_change"]

    wba_settings = [
        ("balanced", 1.0, 1.0),
        ("self_focused", 1.5, 0.5),
        ("socially_influenced", 0.5, 2.0),
    ]

    for label, w_direct, w_social in wba_settings:
        wba_results = wba_predict(scenarios, w_direct=w_direct, w_social=w_social)
        scenario_table[f"WBA_{label}_predicted_second_rating_continuous"] = (
            wba_results["predicted_second_rating_continuous"]
        )
        scenario_table[f"WBA_{label}_predicted_second_rating_rounded"] = (
            wba_results["predicted_second_rating_rounded"]
        )
        scenario_table[f"WBA_{label}_predicted_change"] = wba_results["predicted_change"]

    return scenario_table


def print_scenario_table(df: pd.DataFrame) -> None:
    """Print a rounded version of the scenario table for terminal readability."""
    display_df = df.copy()
    float_columns = display_df.select_dtypes(include=["float64", "float32"]).columns
    display_df[float_columns] = display_df[float_columns].round(3)

    print("=" * 120)
    print("Hand-picked scenario predictions")
    print("=" * 120)
    print(display_df.to_string(index=False))


def save_predictions(df: pd.DataFrame, output_path: Path) -> None:
    """Save the full scenario table to CSV."""
    df.to_csv(output_path, index=False)


def save_continuous_rating_plot(df: pd.DataFrame, output_path: Path) -> None:
    """Plot continuous predicted second ratings across scenarios."""
    labels = df["scenario"]
    x = np.arange(len(labels))
    width = 0.18

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - 1.5 * width, df["SBA_predicted_second_rating_continuous"], width, label="SBA")
    ax.bar(
        x - 0.5 * width,
        df["WBA_balanced_predicted_second_rating_continuous"],
        width,
        label="WBA balanced",
    )
    ax.bar(
        x + 0.5 * width,
        df["WBA_self_focused_predicted_second_rating_continuous"],
        width,
        label="WBA self-focused",
    )
    ax.bar(
        x + 1.5 * width,
        df["WBA_socially_influenced_predicted_second_rating_continuous"],
        width,
        label="WBA socially influenced",
    )

    ax.set_title("Predicted Second Rating Across Hand-Picked Scenarios")
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Predicted second rating (continuous)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(RATING_MIN - 0.1, RATING_MAX + 0.1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_predicted_change_plot(df: pd.DataFrame, output_path: Path) -> None:
    """Plot predicted change across scenarios."""
    labels = df["scenario"]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        x,
        df["SBA_predicted_change"],
        marker="o",
        linestyle="--",
        linewidth=2.5,
        markersize=7,
        color="black",
        label="SBA",
        zorder=4,
    )
    ax.plot(
        x,
        df["WBA_balanced_predicted_change"],
        marker="s",
        linestyle="-",
        linewidth=2,
        markersize=6,
        color="tab:blue",
        label="WBA balanced",
        zorder=3,
    )
    ax.plot(
        x,
        df["WBA_self_focused_predicted_change"],
        marker="^",
        linestyle="-.",
        linewidth=2,
        markersize=6,
        color="tab:orange",
        label="WBA self-focused",
        zorder=2,
    )
    ax.plot(
        x,
        df["WBA_socially_influenced_predicted_change"],
        marker="D",
        linestyle=":",
        linewidth=2,
        markersize=6,
        color="tab:green",
        label="WBA socially influenced",
        zorder=1,
    )

    ax.axhline(0, color="black", linewidth=1, linestyle="--")
    ax.set_title("Predicted Change Across Hand-Picked Scenarios")
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Predicted change")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.legend()
    fig.text(
        0.5,
        0.01,
        "Note: SBA corresponds to WBA with w_direct = 1 and w_social = 1, so exact overlap is expected.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    """Run the scenario plot workflow."""
    project_root = Path(__file__).resolve().parents[1]
    results_dir = project_root / "results" / "scenarios"
    figures_dir = project_root / "figures" / "scenarios"
    ensure_directories(results_dir, figures_dir)

    scenario_table = build_scenario_table()
    print_scenario_table(scenario_table)

    csv_output_path = results_dir / "handpicked_scenarios_predictions.csv"
    rating_plot_path = figures_dir / "handpicked_scenarios_predicted_second_rating.png"
    change_plot_path = figures_dir / "handpicked_scenarios_predicted_change.png"

    save_predictions(scenario_table, csv_output_path)
    save_continuous_rating_plot(scenario_table, rating_plot_path)
    save_predicted_change_plot(scenario_table, change_plot_path)

    print("\nSaved scenario table to:")
    print(csv_output_path)
    print("\nSaved figures to:")
    print(rating_plot_path)
    print(change_plot_path)


if __name__ == "__main__":
    main()
