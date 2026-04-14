"""Create report-friendly plots for the 2x2 synthetic model-recovery results.

This script reads the saved recovery summary table and posterior summary CSVs,
then creates a small set of clear visualizations for model recovery.

Run from the project root:
    python scripts/08b_plot_model_recovery.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def ensure_directories(*paths: Path) -> None:
    """Create output directories if they do not already exist."""
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def load_recovery_summary(csv_path: Path) -> pd.DataFrame:
    """Load the compact recovery summary table."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Recovery summary not found: {csv_path}")
    return pd.read_csv(csv_path)


def load_parameter_summary(csv_path: Path) -> pd.DataFrame:
    """Load a posterior summary CSV indexed by parameter name."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Posterior summary not found: {csv_path}")
    return pd.read_csv(csv_path, index_col=0)


def choose_first_available(columns: pd.Index, candidates: list[str]) -> str | None:
    """Return the first matching column name from a candidate list."""
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def extract_point_and_interval(summary_df: pd.DataFrame, parameter: str) -> tuple[float, float | None, float | None]:
    """Extract posterior mean and an interval when available."""
    if parameter not in summary_df.index:
        raise KeyError(f"Parameter '{parameter}' not found in summary.")

    mean_col = choose_first_available(summary_df.columns, ["Mean", "mean"])
    lower_col = choose_first_available(summary_df.columns, ["5%", "2.5%"])
    upper_col = choose_first_available(summary_df.columns, ["95%", "97.5%"])

    mean_value = float(summary_df.loc[parameter, mean_col])
    lower_value = float(summary_df.loc[parameter, lower_col]) if lower_col is not None else None
    upper_value = float(summary_df.loc[parameter, upper_col]) if upper_col is not None else None
    return mean_value, lower_value, upper_value


def build_pretty_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add a short plot label for each recovery cell."""
    result = df.copy()
    label_map = {
        "SBA_data__SBA_fit": "SBA->SBA",
        "SBA_data__WBA_fit": "SBA->WBA",
        "WBA_data__WBA_fit": "WBA->WBA",
        "WBA_data__SBA_fit": "WBA->SBA",
    }
    result["plot_label"] = result["recovery_cell"].map(label_map)
    return result


def save_sigma_comparison_plot(df: pd.DataFrame, output_path: Path) -> None:
    """Create a bar plot comparing estimated sigma across recovery cells."""
    ordered_labels = ["SBA->SBA", "SBA->WBA", "WBA->WBA", "WBA->SBA"]
    plot_df = df.set_index("plot_label").loc[ordered_labels].reset_index()

    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = ["#56719a", "#7e9cc6", "#b76e5d", "#d49c7d"]
    ax.bar(plot_df["plot_label"], plot_df["estimated_sigma"], color=colors, edgecolor="black")
    ax.set_title("Estimated Sigma Across Model-Recovery Cells")
    ax.set_xlabel("Recovery cell")
    ax.set_ylabel("Estimated sigma")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_sigma_heatmap(df: pd.DataFrame, output_path: Path) -> None:
    """Create a 2x2 heatmap of estimated sigma."""
    heatmap_df = (
        df.assign(
            generating_model=df["generating_dataset"].str.replace("_data", "", regex=False),
        )
        .pivot(index="generating_model", columns="fitted_model", values="estimated_sigma")
        .loc[["SBA", "WBA"], ["SBA", "WBA"]]
    )

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    image = ax.imshow(heatmap_df.values, cmap="YlOrRd", aspect="equal")

    ax.set_xticks(range(len(heatmap_df.columns)))
    ax.set_xticklabels(heatmap_df.columns)
    ax.set_yticks(range(len(heatmap_df.index)))
    ax.set_yticklabels(heatmap_df.index)
    ax.set_title("Estimated Sigma Heatmap")
    ax.set_xlabel("Fitted model")
    ax.set_ylabel("Generating model")

    for row in range(heatmap_df.shape[0]):
        for col in range(heatmap_df.shape[1]):
            ax.text(
                col,
                row,
                f"{heatmap_df.iloc[row, col]:.3f}",
                ha="center",
                va="center",
                color="black",
            )

    fig.colorbar(image, ax=ax, label="Estimated sigma")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_wba_on_sba_parameter_plot(summary_df: pd.DataFrame, output_path: Path) -> None:
    """Plot w_direct and w_social for the WBA fit on SBA synthetic data."""
    parameters = ["w_direct", "w_social"]
    means = []
    lower_errors = []
    upper_errors = []

    for parameter in parameters:
        mean_value, lower_value, upper_value = extract_point_and_interval(summary_df, parameter)
        means.append(mean_value)
        lower_errors.append(mean_value - lower_value if lower_value is not None else 0.0)
        upper_errors.append(upper_value - mean_value if upper_value is not None else 0.0)

    fig, ax = plt.subplots(figsize=(7, 5))
    x_positions = range(len(parameters))
    ax.errorbar(
        x_positions,
        means,
        yerr=[lower_errors, upper_errors],
        fmt="o",
        markersize=8,
        linewidth=2,
        capsize=6,
        color="#3a6f70",
    )
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(parameters)
    ax.set_ylabel("Posterior estimate")
    ax.set_title("WBA Fit on SBA Synthetic Data")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_wba_parameter_comparison_plot(
    summary_sba_data: pd.DataFrame,
    summary_wba_data: pd.DataFrame,
    output_path: Path,
) -> None:
    """Compare WBA parameter estimates across SBA and WBA synthetic datasets."""
    comparison_rows = [
        ("SBA synthetic", summary_sba_data),
        ("WBA synthetic", summary_wba_data),
    ]
    parameters = ["w_direct", "w_social"]
    colors = {"w_direct": "#5f8f71", "w_social": "#b55d4c"}
    offsets = {"w_direct": -0.08, "w_social": 0.08}

    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    for x_index, (dataset_label, summary_df) in enumerate(comparison_rows):
        for parameter in parameters:
            mean_value, lower_value, upper_value = extract_point_and_interval(summary_df, parameter)
            x_position = x_index + offsets[parameter]
            lower_error = mean_value - lower_value if lower_value is not None else 0.0
            upper_error = upper_value - mean_value if upper_value is not None else 0.0

            ax.errorbar(
                x_position,
                mean_value,
                yerr=[[lower_error], [upper_error]],
                fmt="o",
                markersize=8,
                linewidth=2,
                capsize=6,
                color=colors[parameter],
                label=parameter if x_index == 0 else None,
            )

    ax.set_xticks([0, 1])
    ax.set_xticklabels([label for label, _ in comparison_rows])
    ax.set_ylabel("Posterior estimate")
    ax.set_title("WBA Parameter Recovery Across Synthetic Datasets")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    """Read saved summaries and create the model-recovery figures."""
    project_root = Path(__file__).resolve().parents[1]
    results_dir = project_root / "results"
    recovery_summary_path = results_dir / "model_recovery" / "model_recovery_summary.csv"
    fits_dir = results_dir / "fits"
    figures_dir = project_root / "figures" / "model_recovery"
    ensure_directories(figures_dir)

    recovery_df = build_pretty_labels(load_recovery_summary(recovery_summary_path))
    wba_on_sba_summary = load_parameter_summary(fits_dir / "wba_on_sba_synthetic_posterior_summary.csv")
    wba_on_wba_summary = load_parameter_summary(fits_dir / "wba_synthetic_posterior_summary.csv")

    sigma_plot_path = figures_dir / "model_recovery_sigma_comparison.png"
    heatmap_path = figures_dir / "model_recovery_sigma_heatmap.png"
    wba_on_sba_path = figures_dir / "wba_on_sba_parameter_plot.png"
    wba_comparison_path = figures_dir / "wba_parameter_comparison_across_datasets.png"

    save_sigma_comparison_plot(recovery_df, sigma_plot_path)
    save_sigma_heatmap(recovery_df, heatmap_path)
    save_wba_on_sba_parameter_plot(wba_on_sba_summary, wba_on_sba_path)
    save_wba_parameter_comparison_plot(
        summary_sba_data=wba_on_sba_summary,
        summary_wba_data=wba_on_wba_summary,
        output_path=wba_comparison_path,
    )

    print("Saved model recovery figures to:")
    print(sigma_plot_path)
    print(heatmap_path)
    print(wba_on_sba_path)
    print(wba_comparison_path)


if __name__ == "__main__":
    main()
