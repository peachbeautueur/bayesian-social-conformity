"""Summarize the 2x2 synthetic model recovery results.

This script reads the saved posterior summaries from the recovery fits.
It builds a compact comparison table.
It prints the table and saves it for later reporting.

Run from the project root:
    python scripts/08_summarize_model_recovery.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def ensure_directories(*paths: Path) -> None:
    """Create output directories if they do not already exist."""
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def choose_first_available(columns: pd.Index, candidates: list[str]) -> str | None:
    """Return the first matching column name from a candidate list."""
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def extract_parameter_mean(summary_df: pd.DataFrame, parameter: str) -> float | pd.NA:
    """Extract the posterior mean for a parameter when available."""
    if parameter not in summary_df.index:
        return pd.NA

    mean_column = choose_first_available(summary_df.columns, ["Mean", "mean"])
    if mean_column is None:
        return pd.NA

    return summary_df.loc[parameter, mean_column]


def extract_rhat_max(full_summary_df: pd.DataFrame) -> float | pd.NA:
    """Extract the maximum R hat value from a full Stan summary."""
    rhat_column = choose_first_available(full_summary_df.columns, ["R_hat", "Rhat", "r_hat"])
    if rhat_column is None:
        return pd.NA

    rhat_values = pd.to_numeric(full_summary_df[rhat_column], errors="coerce").dropna()
    if rhat_values.empty:
        return pd.NA

    return rhat_values.max()


def build_recovery_row(
    fits_dir: Path,
    cell_name: str,
    generating_dataset: str,
    fitted_model: str,
    posterior_filename: str,
    full_summary_filename: str,
) -> dict:
    """Build one row of the model recovery summary table."""
    posterior_path = fits_dir / posterior_filename
    full_summary_path = fits_dir / full_summary_filename

    posterior_df = pd.read_csv(posterior_path, index_col=0)
    full_summary_df = pd.read_csv(full_summary_path, index_col=0)

    return {
        "recovery_cell": cell_name,
        "generating_dataset": generating_dataset,
        "fitted_model": fitted_model,
        "estimated_sigma": extract_parameter_mean(posterior_df, "sigma"),
        "estimated_w_direct": extract_parameter_mean(posterior_df, "w_direct"),
        "estimated_w_social": extract_parameter_mean(posterior_df, "w_social"),
        "divergences": pd.NA,
        "R_hat_max": extract_rhat_max(full_summary_df),
    }


def build_recovery_summary_table(fits_dir: Path) -> pd.DataFrame:
    """Assemble the four model recovery cells into one compact table."""
    recovery_specs = [
        {
            "cell_name": "SBA_data__SBA_fit",
            "generating_dataset": "SBA_data",
            "fitted_model": "SBA",
            "posterior_filename": "sba_synthetic_posterior_summary.csv",
            "full_summary_filename": "sba_synthetic_full_summary.csv",
        },
        {
            "cell_name": "SBA_data__WBA_fit",
            "generating_dataset": "SBA_data",
            "fitted_model": "WBA",
            "posterior_filename": "wba_on_sba_synthetic_posterior_summary.csv",
            "full_summary_filename": "wba_on_sba_synthetic_full_summary.csv",
        },
        {
            "cell_name": "WBA_data__WBA_fit",
            "generating_dataset": "WBA_data",
            "fitted_model": "WBA",
            "posterior_filename": "wba_synthetic_posterior_summary.csv",
            "full_summary_filename": "wba_synthetic_full_summary.csv",
        },
        {
            "cell_name": "WBA_data__SBA_fit",
            "generating_dataset": "WBA_data",
            "fitted_model": "SBA",
            "posterior_filename": "sba_on_wba_synthetic_posterior_summary.csv",
            "full_summary_filename": "sba_on_wba_synthetic_full_summary.csv",
        },
    ]

    rows = [
        build_recovery_row(
            fits_dir=fits_dir,
            cell_name=spec["cell_name"],
            generating_dataset=spec["generating_dataset"],
            fitted_model=spec["fitted_model"],
            posterior_filename=spec["posterior_filename"],
            full_summary_filename=spec["full_summary_filename"],
        )
        for spec in recovery_specs
    ]
    return pd.DataFrame(rows)


def print_table(df: pd.DataFrame) -> None:
    """Print a rounded version of the summary table."""
    display_df = df.copy()
    numeric_columns = [
        "estimated_sigma",
        "estimated_w_direct",
        "estimated_w_social",
        "R_hat_max",
    ]
    for column in numeric_columns:
        if column in display_df.columns:
            display_df[column] = pd.to_numeric(display_df[column], errors="coerce").round(3)

    print("=" * 100)
    print("Model recovery summary")
    print("=" * 100)
    print(display_df.to_string(index=False))


def print_interpretive_notes(df: pd.DataFrame) -> None:
    """Print a short automatic interpretation when supported by the results."""
    row_wba_on_sba = df.loc[df["recovery_cell"] == "SBA_data__WBA_fit"].iloc[0]
    row_sba_on_sba = df.loc[df["recovery_cell"] == "SBA_data__SBA_fit"].iloc[0]
    row_sba_on_wba = df.loc[df["recovery_cell"] == "WBA_data__SBA_fit"].iloc[0]

    notes: list[str] = []

    w_direct = pd.to_numeric(row_wba_on_sba["estimated_w_direct"], errors="coerce")
    w_social = pd.to_numeric(row_wba_on_sba["estimated_w_social"], errors="coerce")
    if pd.notna(w_direct) and pd.notna(w_social):
        if abs(w_direct - 1.0) < 0.35 and abs(w_social - 1.0) < 0.35:
            notes.append("WBA fit on SBA data recovers weights close to 1, consistent with equal weighting.")
        else:
            notes.append(
                f"WBA fit on SBA data does not land exactly at 1/1, but the estimates "
                f"({w_direct:.3f}, {w_social:.3f}) can still be compared against equal weighting."
            )

    sigma_sba_on_sba = pd.to_numeric(row_sba_on_sba["estimated_sigma"], errors="coerce")
    sigma_sba_on_wba = pd.to_numeric(row_sba_on_wba["estimated_sigma"], errors="coerce")
    if pd.notna(sigma_sba_on_sba) and pd.notna(sigma_sba_on_wba):
        if sigma_sba_on_wba > sigma_sba_on_sba:
            notes.append(
                f"SBA fit on WBA data needs a larger sigma ({sigma_sba_on_wba:.3f}) than "
                f"SBA fit on SBA data ({sigma_sba_on_sba:.3f}), which is the expected recovery pattern."
            )
        else:
            notes.append(
                f"SBA fit on WBA data does not show a larger sigma than SBA fit on SBA data "
                f"({sigma_sba_on_wba:.3f} vs {sigma_sba_on_sba:.3f})."
            )

    if notes:
        print("\nAutomatic notes:")
        for note in notes:
            print(f"- {note}")


def main() -> None:
    """Read the saved fit summaries and build a compact recovery table."""
    project_root = Path(__file__).resolve().parents[1]
    fits_dir = project_root / "results" / "fits"
    output_dir = project_root / "results" / "model_recovery"
    ensure_directories(output_dir)

    summary_df = build_recovery_summary_table(fits_dir)
    output_path = output_dir / "model_recovery_summary.csv"
    summary_df.to_csv(output_path, index=False)

    print_table(summary_df)
    print_interpretive_notes(summary_df)

    print("\nSaved summary to:")
    print(output_path)


if __name__ == "__main__":
    main()
