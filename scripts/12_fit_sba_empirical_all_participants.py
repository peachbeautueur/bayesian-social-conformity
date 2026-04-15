"""Fit the Stan SBA model separately to all empirical participants.

This script runs a no pooling analysis for the Simonsen dataset.
Each participant is fit separately with the SBA Stan model.
It saves a compact summary table for later comparison with WBA fits.

Run from the project root:
    python scripts/12_fit_sba_empirical_all_participants.py
"""

from __future__ import annotations

import locale
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
from cmdstanpy import CmdStanModel
import cmdstanpy.compilation as cmdstan_compilation
import cmdstanpy.utils.command as cmdstan_command


ALPHA0 = 1.0
BETA0 = 1.0
N_DIRECT = 7.0
N_SOCIAL = 7.0
SEED = 123
CHAINS = 4
PARALLEL_CHAINS = 4
ITER_WARMUP = 1000
ITER_SAMPLING = 1000


def ensure_directories(*paths: Path) -> None:
    """Create output directories if they do not already exist."""
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def load_empirical_data(csv_path: Path) -> pd.DataFrame:
    """Load the Simonsen empirical dataset."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Empirical dataset not found: {csv_path}")
    return pd.read_csv(csv_path)


def get_sorted_participant_ids(df: pd.DataFrame) -> list[int]:
    """Return all unique participant IDs in sorted order."""
    if "ID" not in df.columns:
        raise ValueError("The empirical dataset does not contain an 'ID' column.")
    participant_ids = sorted(df["ID"].dropna().unique().tolist())
    if not participant_ids:
        raise ValueError("No participant IDs were found in the empirical dataset.")
    return participant_ids


def patch_cmdstanpy_windows_encoding() -> None:
    """Patch cmdstanpy command decoding for Windows compiler output."""
    preferred_encoding = locale.getpreferredencoding(False) or "utf-8"

    def do_command_patched(cmd, cwd=None, *, fd_out=sys.stdout, pbar=None) -> None:
        cmdstan_command.get_logger().debug("cmd: %s\ncwd: %s", " ".join(cmd), cwd)
        try:
            with cmdstan_command.pushd(cwd if cwd is not None else "."):
                proc = subprocess.Popen(
                    cmd,
                    bufsize=1,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    env=os.environ,
                    universal_newlines=True,
                    encoding=preferred_encoding,
                    errors="replace",
                )
                while proc.poll() is None:
                    if proc.stdout is not None:
                        line = proc.stdout.readline()
                        if fd_out is not None:
                            fd_out.write(line)
                        if pbar is not None:
                            pbar(line.strip())

                stdout, _ = proc.communicate()
                if stdout and len(stdout) > 0:
                    if fd_out is not None:
                        fd_out.write(stdout)
                    if pbar is not None:
                        pbar(stdout.strip())

                if proc.returncode != 0:
                    serror = ""
                    try:
                        serror = os.strerror(proc.returncode)
                    except (ArithmeticError, ValueError):
                        pass
                    msg = "Command {}\n\texited with code '{}' {}".format(
                        cmd,
                        proc.returncode,
                        serror,
                    )
                    raise RuntimeError(msg)
        except OSError as exc:
            msg = "Command: {}\nfailed with error {}\n".format(cmd, str(exc))
            raise RuntimeError(msg) from exc

    cmdstan_command.do_command = do_command_patched
    cmdstan_compilation.do_command = do_command_patched


def prepare_stan_data(df: pd.DataFrame) -> dict:
    """Prepare the data dictionary expected by the Stan model."""
    required_columns = ["FirstRating", "GroupRating", "SecondRating"]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns for Stan fitting: {missing_columns}")

    return {
        "N": len(df),
        "first_rating": df["FirstRating"].astype(int).tolist(),
        "group_rating": df["GroupRating"].astype(int).tolist(),
        "second_rating": df["SecondRating"].astype(float).tolist(),
        "alpha0": ALPHA0,
        "beta0": BETA0,
        "n_direct": N_DIRECT,
        "n_social": N_SOCIAL,
    }


def choose_first_available(columns: pd.Index, candidates: list[str]) -> str | None:
    """Return the first matching column name from a candidate list."""
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def extract_parameter_mean(summary: pd.DataFrame, parameter: str) -> float | pd.NA:
    """Extract the posterior mean for one parameter when available."""
    if parameter not in summary.index:
        return pd.NA
    mean_col = choose_first_available(summary.columns, ["Mean", "mean"])
    if mean_col is None:
        return pd.NA
    return float(summary.loc[parameter, mean_col])


def extract_rhat_max(summary: pd.DataFrame) -> float | pd.NA:
    """Extract the maximum R hat value from a full Stan summary."""
    rhat_col = choose_first_available(summary.columns, ["R_hat", "Rhat", "r_hat"])
    if rhat_col is None:
        return pd.NA
    rhat_values = pd.to_numeric(summary[rhat_col], errors="coerce").dropna()
    if rhat_values.empty:
        return pd.NA
    return float(rhat_values.max())


def count_divergences(fit) -> int:
    """Count total divergent transitions across all post warmup draws."""
    sampler_diagnostics = fit.method_variables()
    return int(sampler_diagnostics["divergent__"].sum())


def fit_one_participant(
    model: CmdStanModel,
    participant_id: int,
    participant_df: pd.DataFrame,
    results_dir: Path,
) -> dict:
    """Fit one participant and return a compact summary row."""
    participant_output_dir = results_dir / "sba_per_participant"
    ensure_directories(participant_output_dir)

    stan_data = prepare_stan_data(participant_df)
    fit = model.sample(
        data=stan_data,
        seed=SEED + int(participant_id),
        chains=CHAINS,
        parallel_chains=PARALLEL_CHAINS,
        iter_warmup=ITER_WARMUP,
        iter_sampling=ITER_SAMPLING,
        output_dir=str(participant_output_dir),
        show_progress=True,
    )

    full_summary = fit.summary()
    full_summary_path = participant_output_dir / f"sba_empirical_participant_{participant_id}_full_summary.csv"
    full_summary.to_csv(full_summary_path)

    return {
        "participant_id": int(participant_id),
        "n_trials": int(len(participant_df)),
        "posterior_mean_sigma": extract_parameter_mean(full_summary, "sigma"),
        "divergences": count_divergences(fit),
        "R_hat_max": extract_rhat_max(full_summary),
        "status": "success",
        "error_message": pd.NA,
    }


def main() -> None:
    """Fit the SBA Stan model separately to all empirical participants."""
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "Simonsen_clean.csv"
    stan_file = project_root / "stan" / "sba_model.stan"
    results_dir = project_root / "results" / "fits" / "empirical"
    ensure_directories(results_dir)

    patch_cmdstanpy_windows_encoding()

    df = load_empirical_data(data_path)
    participant_ids = get_sorted_participant_ids(df)
    model = CmdStanModel(stan_file=str(stan_file))

    summary_rows: list[dict] = []

    print(f"Found {len(participant_ids)} participants. Starting no-pooling SBA fits...")

    for index, participant_id in enumerate(participant_ids, start=1):
        participant_df = df.loc[df["ID"] == participant_id].copy()
        print(f"\n[{index}/{len(participant_ids)}] Fitting participant {participant_id} ({len(participant_df)} trials)")

        try:
            summary_row = fit_one_participant(
                model=model,
                participant_id=participant_id,
                participant_df=participant_df,
                results_dir=results_dir,
            )
            print(
                "Completed participant "
                f"{participant_id}: sigma={summary_row['posterior_mean_sigma']:.3f}, "
                f"divergences={summary_row['divergences']}"
            )
        except Exception as exc:  # noqa: BLE001
            summary_row = {
                "participant_id": int(participant_id),
                "n_trials": int(len(participant_df)),
                "posterior_mean_sigma": pd.NA,
                "divergences": pd.NA,
                "R_hat_max": pd.NA,
                "status": "failed",
                "error_message": str(exc),
            }
            print(f"Failed participant {participant_id}: {exc}")

        summary_rows.append(summary_row)

    summary_df = pd.DataFrame(summary_rows)
    output_path = results_dir / "sba_empirical_all_participants_summary.csv"
    summary_df.to_csv(output_path, index=False)

    print("\nFinished all participant fits.")
    print(f"Saved summary table to: {output_path}")
    print("\nSummary preview:")
    print(summary_df.head().to_string(index=False))


if __name__ == "__main__":
    main()
