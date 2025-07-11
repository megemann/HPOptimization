# Optimization Results Dashboard

A Plotly Dash application for visualizing hyperparameter optimization results from any directory in the WIP folder structure.

## Features

- **Multi-format support**: Automatically detects and handles different result formats:
  - Phase 2 format (CaliforniaHousing/UCILetter): `results_1.json`, `results_2.json`, `results_3.json`
  - Branin format: `*_results.json` (e.g., `50_results.json`)
- **Interactive controls**:
  - Select multiple directories to compare
  - Set Y-axis limits (min/max)
  - Set X-axis limit (time max)
  - Custom plot titles
- **Automatic averaging**: Averages across multiple runs when available
- **Smart line naming**: Uses directory path as line name

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the dashboard:
```bash
python dashboard.py
```

2. Open your browser to `http://localhost:8050`

3. Use the interface:
   - **Select Directories**: Choose one or more result directories from the dropdown
   - **Plot Title**: Enter a custom title for your plot
   - **Y Min/Max**: Set Y-axis limits (leave blank for auto)
   - **X Max**: Set maximum time to display (leave blank for all data)
   - **Update Plot**: Click to refresh the plot with new settings

## Data Format Support

### Phase 2 Format (CaliforniaHousing/UCILetter)
- Files: `results_1.json`, `results_2.json`, `results_3.json`
- Expected structure: List of trial results with `trial_end`, and one of `mse`, `accuracy`, or `loss`
- Automatically averages across multiple runs

### Branin Format
- Files: `*_results.json` (e.g., `50_results.json`)
- Expected structure: Dictionary with method names as keys (e.g., `random_results`, `tpe_results`)
- Each method contains runs with trials having `timestamp` and `value`

## Examples

- Compare TPE startup trials: Select multiple directories like `TPE/n_startup_trials/n_startup_trials=0`, `TPE/n_startup_trials/n_startup_trials=5`, etc.
- Compare HYPERBAND configurations: Select `HYPERBAND/mr=3/n=2`, `HYPERBAND/mr=3/n=3`, etc.
- View Branin results: Select directories containing `*_results.json` files

The dashboard will automatically detect the format and plot the best-so-far values over time, just like your matplotlib examples. 