# Bayesian Social Conformity

This project studies social conformity in trustworthiness ratings with simple Bayesian models.
It compares two agents:

- `SBA`: Simple Bayesian Agent
- `WBA`: Weighted Bayesian Agent

The project includes:

- data exploration
- hand built scenario checks
- synthetic data simulation
- Stan fitting
- model recovery
- empirical model comparison
- posterior predictive checks

## Research Idea

Each trial has:

- `FirstRating`: the participant's first trust rating
- `GroupRating`: the social information
- `SecondRating`: the updated trust rating

The models map ratings from `1` to `8` onto probabilities in `[0, 1]` with:

`p = (rating - 1) / 7`

The models then treat these values as fixed evidence in a beta update.

## Models

### SBA

SBA takes direct and social information at face value.
It has no weight parameters.

### WBA

WBA uses the same beta update structure.
It adds two positive weights:

- `w_direct`
- `w_social`

This allows the model to give different importance to direct and social information.

## Data

The project uses three CSV files in `data/`:

- `cogsci_clean.csv`
- `sc_df_clean.csv`
- `Simonsen_clean.csv`

The main empirical analysis uses:

- `data/Simonsen_clean.csv`

Expected columns:

- `ID`
- `FaceID`
- `FirstRating`
- `GroupRating`
- `SecondRating`

## Project Structure

```text
data/        cleaned datasets
scripts/     Python workflow scripts
stan/        Stan model files
results/     saved tables and model summaries
figures/     saved plots
report/      report material
```

## Script Order

The scripts are numbered to match the workflow.

### Exploration and agent definition

- `scripts/00_load_and_explore.py`
  Load all datasets. Save basic summaries and plots.
- `scripts/01_define_sba_wba_agents.py`
  Define the SBA and WBA prototype agents.
- `scripts/02_visualize_handpicked_scenarios.py`
  Compare SBA and WBA on five selected scenarios.

### Synthetic data and recovery

- `scripts/03_simulate_synthetic_data.py`
  Simulate SBA and WBA synthetic data.
- `scripts/04_fit_wba_synthetic.py`
  Fit WBA to WBA synthetic data.
- `scripts/05_fit_sba_synthetic.py`
  Fit SBA to SBA synthetic data.
- `scripts/06_fit_wba_to_sba_synthetic.py`
  Fit WBA to SBA synthetic data.
- `scripts/07_fit_sba_to_wba_synthetic.py`
  Fit SBA to WBA synthetic data.
- `scripts/08_summarize_model_recovery.py`
  Summarize the 2x2 recovery table.
- `scripts/08b_plot_model_recovery.py`
  Plot recovery results.

### Empirical exploration and fitting

- `scripts/08c_plot_summaries_and_exploration.py`
  Create empirical exploration figures.
- `scripts/09_fit_wba_empirical_single_participant.py`
  Fit WBA to one participant.
- `scripts/10_fit_sba_empirical_single_participant.py`
  Fit SBA to the same participant.
- `scripts/11_fit_wba_empirical_all_participants.py`
  Fit WBA to all participants with no pooling.
- `scripts/12_fit_sba_empirical_all_participants.py`
  Fit SBA to all participants with no pooling.
- `scripts/13_compare_empirical_models.py`
  Compare empirical SBA and WBA fit summaries.
- `scripts/14_compare_models_empirical_single_participant.py`
  Formal model comparison for participant `201`.
- `scripts/15_compare_models_empirical_all_participants.py`
  Formal model comparison across all participants.

### Posterior predictive checks and prior posterior plots

- `scripts/16_ppc_empirical_aggregate.py`
  Aggregate empirical posterior predictive checks.
- `scripts/17_ppc_participant_level_mean_change.py`
  Participant level PPC for mean rating change.
- `scripts/18_prior_posterior_updates_wba_empirical.py`
  Prior posterior summary for empirical WBA fits.

## How To Run

Run scripts from the project root.

Example:

```bash
python scripts/00_load_and_explore.py
```


## Main Output Locations

### Tables

- `results/model_recovery/model_recovery_summary.csv`
- `results/fits/empirical/empirical_model_comparison_summary.csv`
- `results/fits/empirical/empirical_all_participants_model_comparison.csv`
- `results/ppc/participant_level_mean_change_ppc_summary.csv`
- `results/prior_posterior/wba_empirical_prior_posterior_summary.csv`

### Figures

- `figures/model_recovery/`
- `figures/exploration_report/`
- `figures/empirical/`
- `figures/empirical_model_comparison/`
- `figures/ppc/`
- `figures/prior_posterior/`

## Notes On Tracked Files

The repository only keeps:

- scripts
- Stan model files
- compact summary tables
- report figures


## Dependencies

Main Python packages used in this project:

- `pandas`
- `numpy`
- `matplotlib`
- `cmdstanpy`

Some scripts also use:

- `pathlib`
- `arviz` if available

## Current Empirical Result

In the no pooling empirical comparison on `Simonsen_clean.csv`, WBA performed better than SBA for all `40` participants by WAIC.
