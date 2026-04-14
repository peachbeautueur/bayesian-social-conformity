"""Fit the first Stan WBA model to synthetic WBA data.

This script loads the synthetic WBA dataset, prepares Stan inputs, fits the
lecture-inspired Weighted Bayesian Agent model with cmdstanpy, and saves basic
fit outputs for inspection.

Run from the project root:
    python scripts/04_fit_wba_synthetic.py
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


def load_synthetic_wba_data(csv_path: Path) -> pd.DataFrame:
    """Load the synthetic WBA dataset."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Synthetic WBA data not found: {csv_path}")
    return pd.read_csv(csv_path)


def patch_cmdstanpy_windows_encoding() -> None:
    """Patch cmdstanpy command decoding for Windows compiler output.

    Some Windows toolchains emit non-UTF-8 console text during compilation.
    cmdstanpy 1.3.0 reads subprocess output with text mode defaults, which can
    trigger UnicodeDecodeError. This patch keeps the logic the same but decodes
    using the local preferred encoding and replaces problematic bytes.
    """
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


def print_available_summary_columns(summary: pd.DataFrame) -> None:
    """Print the available cmdstanpy summary columns for debugging."""
    print("\nAvailable summary columns:")
    print(", ".join(summary.columns.astype(str).tolist()))


def choose_available_columns(summary: pd.DataFrame) -> list[str]:
    """Choose a robust subset of summary columns across cmdstanpy versions."""
    preferred_column_groups = [
        ["Mean"],
        ["StdDev", "SD"],
        ["5%", "2.5%"],
        ["50%"],
        ["95%", "97.5%"],
        ["R_hat"],
        ["ESS_bulk"],
    ]

    selected_columns: list[str] = []
    for candidates in preferred_column_groups:
        for column in candidates:
            if column in summary.columns:
                selected_columns.append(column)
                break

    return selected_columns


def summarize_posterior(summary: pd.DataFrame) -> pd.DataFrame:
    """Extract a concise posterior summary for the main parameters."""
    parameter_rows = [row for row in ["w_direct", "w_social", "sigma"] if row in summary.index]
    selected_columns = choose_available_columns(summary)

    if not parameter_rows:
        raise ValueError("Parameter rows w_direct, w_social, and sigma were not found in the Stan summary.")
    if not selected_columns:
        raise ValueError("No expected summary columns were found in the Stan summary output.")

    return summary.loc[parameter_rows, selected_columns]


def count_divergences(fit) -> int:
    """Count total divergent transitions across all post-warmup draws."""
    sampler_diagnostics = fit.method_variables()
    return int(sampler_diagnostics["divergent__"].sum())


def summarize_rhat(fit) -> tuple[float, float]:
    """Return the max and median R-hat values when available."""
    summary = fit.summary()
    if "R_hat" not in summary.columns:
        return float("nan"), float("nan")
    rhat_series = summary["R_hat"].dropna()
    if rhat_series.empty:
        return float("nan"), float("nan")
    return float(rhat_series.median()), float(rhat_series.max())


def main() -> None:
    """Fit the WBA Stan model to the synthetic WBA data."""
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "results" / "simulated_data" / "simulated_wba_data.csv"
    stan_file = project_root / "stan" / "wba_model.stan"
    results_dir = project_root / "results" / "fits"
    ensure_directories(results_dir)

    df = load_synthetic_wba_data(data_path)
    stan_data = prepare_stan_data(df)
    patch_cmdstanpy_windows_encoding()

    try:
        model = CmdStanModel(stan_file=str(stan_file))
        fit = model.sample(
            data=stan_data,
            seed=SEED,
            chains=CHAINS,
            parallel_chains=PARALLEL_CHAINS,
            iter_warmup=ITER_WARMUP,
            iter_sampling=ITER_SAMPLING,
            output_dir=str(results_dir),
            show_progress=True,
        )
    except ValueError as exc:
        raise RuntimeError(
            "Stan model compilation failed. cmdstanpy is installed, but the C++ "
            "toolchain appears unavailable. On Windows, install and configure a "
            "working g++ toolchain for CmdStan, then rerun this script."
        ) from exc

    full_summary = fit.summary()
    print_available_summary_columns(full_summary)
    posterior_summary = summarize_posterior(full_summary)
    posterior_summary_path = results_dir / "wba_synthetic_posterior_summary.csv"
    full_summary_path = results_dir / "wba_synthetic_full_summary.csv"

    posterior_summary.to_csv(posterior_summary_path)
    full_summary.to_csv(full_summary_path)

    divergences = count_divergences(fit)
    rhat_median, rhat_max = summarize_rhat(fit)

    print("\n" + "=" * 80)
    print("Posterior summary for synthetic WBA fit")
    print("=" * 80)
    print(posterior_summary.round(3).to_string())

    print("\nDiagnostics")
    print(f"Divergences: {divergences}")
    print(f"R-hat median: {rhat_median:.3f}")
    print(f"R-hat max: {rhat_max:.3f}")

    print("\nSaved outputs to:")
    print(results_dir)
    print(posterior_summary_path)
    print(full_summary_path)


if __name__ == "__main__":
    main()
