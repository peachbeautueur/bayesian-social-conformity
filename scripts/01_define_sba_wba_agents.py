"""Define proof-of-concept SBA and WBA agents for 1-8 rating updates.

This script adapts the lecture's beta-update logic from binary evidence tasks
to trustworthiness ratings on a 1-8 scale. Ratings are first mapped to
probabilities, then treated as pseudo-binomial evidence with fixed evidence
strength. The goal here is transparency and forward simulation, not model
fitting.

Run from the project root:
    python scripts/01_define_sba_wba_agents.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd


ALPHA0 = 1.0
BETA0 = 1.0
N_DIRECT = 7
N_SOCIAL = 7
RATING_MIN = 1
RATING_MAX = 8


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
    """Predict updated ratings under the Simple Bayesian Agent.

    In the SBA, direct evidence and social evidence are both taken at face
    value, with no additional weighting parameters.
    """
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

    result["SBA_predicted_second_rating_continuous"] = predicted_continuous
    result["SBA_predicted_second_rating_rounded"] = predicted_rounded
    result["SBA_predicted_change"] = predicted_continuous - result["FirstRating"]

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
    """Predict updated ratings under the Weighted Bayesian Agent.

    In the WBA, the same beta-update structure is used, but direct and social
    evidence can carry different weights.
    """
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

    result["WBA_predicted_second_rating_continuous"] = predicted_continuous
    result["WBA_predicted_second_rating_rounded"] = predicted_rounded
    result["WBA_predicted_change"] = predicted_continuous - result["FirstRating"]
    result["w_direct"] = w_direct
    result["w_social"] = w_social

    return result


def build_example_trials() -> pd.DataFrame:
    """Create a small set of transparent example trials."""
    return pd.DataFrame(
        {
            "TrialLabel": [
                "low self, high group",
                "high self, low group",
                "equal ratings",
                "moderate disagreement up",
                "moderate disagreement down",
            ],
            "FirstRating": [2, 7, 4, 3, 6],
            "GroupRating": [7, 2, 4, 5, 4],
            "SecondRating": [np.nan, np.nan, np.nan, np.nan, np.nan],
        }
    )


def format_comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    """Select and round columns for neat console printing."""
    table = df[
        [
            "TrialLabel",
            "FirstRating",
            "GroupRating",
            "SBA_predicted_second_rating_continuous",
            "SBA_predicted_second_rating_rounded",
            "SBA_predicted_change",
            "WBA_predicted_second_rating_continuous",
            "WBA_predicted_second_rating_rounded",
            "WBA_predicted_change",
        ]
    ].copy()

    rounded_columns = [
        "SBA_predicted_second_rating_continuous",
        "SBA_predicted_change",
        "WBA_predicted_second_rating_continuous",
        "WBA_predicted_change",
    ]
    table[rounded_columns] = table[rounded_columns].round(3)
    return table


def print_model_notes() -> None:
    """Print a short explanation of the prototype assumptions."""
    print("=" * 80)
    print("Prototype SBA / WBA definitions for trustworthiness ratings")
    print("=" * 80)
    print("1. Ratings on the 1-8 scale are mapped to probabilities using p = (rating - 1) / 7.")
    print("2. Each rating is treated as pseudo-binomial evidence with fixed evidence size n = 7.")
    print("3. The prior is Beta(alpha0=1, beta0=1).")
    print("4. SBA trusts both sources equally; WBA allows separate weights on direct and social evidence.")
    print("5. The posterior mean is mapped back to the 1-8 rating scale for prediction.\n")


def main() -> None:
    """Run a simple forward-simulation demonstration for SBA and WBA."""
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", None)

    example_df = build_example_trials()

    print_model_notes()
    print("Example trials:")
    print(example_df[["TrialLabel", "FirstRating", "GroupRating"]].to_string(index=False))

    sba_results = sba_predict(example_df)

    wba_settings = [
        ("balanced", 1.0, 1.0),
        ("self-focused", 1.5, 0.5),
        ("socially influenced", 0.5, 2.0),
    ]

    print("\n" + "=" * 80)
    print("SBA predictions")
    print("=" * 80)
    sba_table = sba_results[
        [
            "TrialLabel",
            "FirstRating",
            "GroupRating",
            "SBA_predicted_second_rating_continuous",
            "SBA_predicted_second_rating_rounded",
            "SBA_predicted_change",
        ]
    ].copy()
    sba_table[
        ["SBA_predicted_second_rating_continuous", "SBA_predicted_change"]
    ] = sba_table[
        ["SBA_predicted_second_rating_continuous", "SBA_predicted_change"]
    ].round(3)
    print(sba_table.to_string(index=False))

    for label, w_direct, w_social in wba_settings:
        print("\n" + "=" * 80)
        print(f"WBA predictions: {label} (w_direct={w_direct}, w_social={w_social})")
        print("=" * 80)

        wba_results = wba_predict(example_df, w_direct=w_direct, w_social=w_social)
        combined = sba_results.merge(
            wba_results[
                [
                    "TrialLabel",
                    "WBA_predicted_second_rating_continuous",
                    "WBA_predicted_second_rating_rounded",
                    "WBA_predicted_change",
                ]
            ],
            on="TrialLabel",
            how="left",
        )
        comparison_table = format_comparison_table(combined)
        print(comparison_table.to_string(index=False))


if __name__ == "__main__":
    main()
