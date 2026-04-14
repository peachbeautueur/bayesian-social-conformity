"""Simulate synthetic participant-level data from SBA and WBA agents.

This script reuses the same lecture-inspired beta-update logic defined in the
prototype SBA and WBA agent script. It creates transparent forward-simulated
datasets that can later be used for model recovery and Stan fitting.

Run from the project root:
    python scripts/03_simulate_synthetic_data.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ALPHA0 = 1.0
BETA0 = 1.0
N_DIRECT = 7
N_SOCIAL = 7
RATING_MIN = 1
RATING_MAX = 8
N_PARTICIPANTS = 20
N_TRIALS = 80
SEED = 123
TRUE_W_DIRECT = 0.5
TRUE_W_SOCIAL = 2.0
SIGMA_OBS = 0.5


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

    result["SecondRating_continuous"] = probability_to_rating(posterior_mean)
    result["change"] = result["SecondRating_continuous"] - result["FirstRating"]

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

    result["SecondRating_continuous"] = probability_to_rating(posterior_mean)
    result["change"] = result["SecondRating_continuous"] - result["FirstRating"]

    return result


def build_base_trials(
    rng: np.random.Generator,
    generating_model: str,
    participant_id_start: int,
    n_participants: int = N_PARTICIPANTS,
    n_trials: int = N_TRIALS,
) -> pd.DataFrame:
    """Create participant-by-trial rows with random first and group ratings."""
    participant_ids = np.repeat(
        np.arange(participant_id_start, participant_id_start + n_participants),
        n_trials,
    )
    trial_ids = np.tile(np.arange(1, n_trials + 1), n_participants)
    n_rows = n_participants * n_trials

    df = pd.DataFrame(
        {
            "generating_model": generating_model,
            "participant_id": participant_ids,
            "trial": trial_ids,
            "FirstRating": rng.integers(RATING_MIN, RATING_MAX + 1, size=n_rows),
            "GroupRating": rng.integers(RATING_MIN, RATING_MAX + 1, size=n_rows),
        }
    )
    df["social_gap"] = df["GroupRating"] - df["FirstRating"]
    return df


def add_observation_noise(
    df: pd.DataFrame,
    rng: np.random.Generator,
    sigma_obs: float = SIGMA_OBS,
) -> pd.DataFrame:
    """Add Gaussian observation noise before discretizing the observed rating."""
    result = df.copy()
    noise = rng.normal(loc=0.0, scale=sigma_obs, size=len(result))
    result["SecondRating_noisy_continuous"] = result["SecondRating_continuous"] + noise
    result["SecondRating"] = np.clip(
        np.rint(result["SecondRating_noisy_continuous"]),
        RATING_MIN,
        RATING_MAX,
    ).astype(int)
    result["observed_change"] = result["SecondRating"] - result["FirstRating"]
    return result


def finalize_columns(
    df: pd.DataFrame,
    generating_model: str,
    true_w_direct: float,
    true_w_social: float,
) -> pd.DataFrame:
    """Keep columns in the requested order."""
    result = df.copy()
    result["generating_model"] = generating_model
    result["true_w_direct"] = true_w_direct
    result["true_w_social"] = true_w_social

    ordered_columns = [
        "generating_model",
        "participant_id",
        "trial",
        "FirstRating",
        "GroupRating",
        "SecondRating_continuous",
        "SecondRating_noisy_continuous",
        "SecondRating",
        "observed_change",
        "social_gap",
        "change",
        "true_w_direct",
        "true_w_social",
    ]
    return result[ordered_columns]


def simulate_sba_data(rng: np.random.Generator) -> pd.DataFrame:
    """Simulate a dataset generated by the SBA."""
    base_df = build_base_trials(
        rng=rng,
        generating_model="SBA",
        participant_id_start=1,
    )
    predicted_df = sba_predict(base_df)
    noisy_df = add_observation_noise(predicted_df, rng=rng)
    return finalize_columns(
        noisy_df,
        generating_model="SBA",
        true_w_direct=np.nan,
        true_w_social=np.nan,
    )


def simulate_wba_data(rng: np.random.Generator) -> pd.DataFrame:
    """Simulate a dataset generated by the WBA with fixed true weights."""
    base_df = build_base_trials(
        rng=rng,
        generating_model="WBA",
        participant_id_start=N_PARTICIPANTS + 1,
    )
    predicted_df = wba_predict(
        base_df,
        w_direct=TRUE_W_DIRECT,
        w_social=TRUE_W_SOCIAL,
    )
    noisy_df = add_observation_noise(predicted_df, rng=rng)
    return finalize_columns(
        noisy_df,
        generating_model="WBA",
        true_w_direct=TRUE_W_DIRECT,
        true_w_social=TRUE_W_SOCIAL,
    )


def print_summary(df: pd.DataFrame, model_name: str) -> None:
    """Print a concise summary for one simulated dataset."""
    n_participants = df["participant_id"].nunique()
    n_trials = len(df)
    second_rating_mean = df["SecondRating"].mean()
    second_rating_sd = df["SecondRating"].std(ddof=1)
    noisy_continuous_mean = df["SecondRating_noisy_continuous"].mean()
    noisy_continuous_sd = df["SecondRating_noisy_continuous"].std(ddof=1)
    change_mean = df["change"].mean()
    change_sd = df["change"].std(ddof=1)

    print("\n" + "=" * 80)
    print(f"Summary for {model_name} generating model")
    print("=" * 80)
    print(f"Participants: {n_participants}")
    print(f"Trials: {n_trials}")
    print(
        f"SecondRating_noisy_continuous mean (sd): "
        f"{noisy_continuous_mean:.3f} ({noisy_continuous_sd:.3f})"
    )
    print(f"SecondRating mean (sd): {second_rating_mean:.3f} ({second_rating_sd:.3f})")
    print(f"Change mean (sd): {change_mean:.3f} ({change_sd:.3f})")


def main() -> None:
    """Simulate SBA and WBA datasets and save them to disk."""
    project_root = Path(__file__).resolve().parents[1]
    results_dir = project_root / "results" / "simulated_data"
    ensure_directories(results_dir)

    rng = np.random.default_rng(SEED)

    sba_df = simulate_sba_data(rng)
    wba_df = simulate_wba_data(rng)
    combined_df = pd.concat([sba_df, wba_df], ignore_index=True)

    sba_output_path = results_dir / "simulated_sba_data.csv"
    wba_output_path = results_dir / "simulated_wba_data.csv"
    combined_output_path = results_dir / "simulated_combined_data.csv"

    sba_df.to_csv(sba_output_path, index=False)
    wba_df.to_csv(wba_output_path, index=False)
    combined_df.to_csv(combined_output_path, index=False)

    print_summary(sba_df, model_name="SBA")
    print_summary(wba_df, model_name="WBA")

    print("\nSaved simulated datasets to:")
    print(sba_output_path)
    print(wba_output_path)
    print(combined_output_path)


if __name__ == "__main__":
    main()
