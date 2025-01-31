# Hyperparameter Optimization Examples

This directory contains examples of different automated hyperparameter optimization (HPO) approaches using popular frameworks:

- Scikit-learn's RandomizedSearchCV and ParameterSampler
- Optuna
- Keras Tuner

## Setup

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install the required packages:

```bash
pip install -r requirements.txt
```

## Running the Examples

Each example can be run independently:

### Scikit-learn Example

```bash
python scikit-learn/main.py
```
This demonstrates both RandomizedSearchCV and ParameterSampler approaches in one run.

### Optuna Example

```bash
python optuna_example.py
```
Shows Optuna's trial-based optimization approach with dynamic parameter suggestions.

### Keras Tuner Example

```bash
python keras_tuner_example.py
```
Demonstrates Keras Tuner's Hyperband algorithm with early stopping.

## Example Structure

- `scikit-learn/`: Contains scikit-learn based approaches
  - `main.py`: Main script to run both scikit-learn examples
  - `model_definition.py`: Model architecture definition
  - `parameter_sampler_example.py`: ParameterSampler implementation
  - `randomized_search_example.py`: RandomizedSearchCV implementation

- `optuna_example.py`: Complete Optuna implementation
- `keras_tuner_example.py`: Complete Keras Tuner implementation
- `generate_dummy_data.py`: Generates dummy data for examples
- `base_model.py`: Base model architecture

## Notes

- All examples use the same model architecture and hyperparameter space for comparison
- Each framework has its own advantages:
  - Scikit-learn: Simple integration with sklearn ecosystem
  - Optuna: Flexible parameter definition and pruning
  - Keras Tuner: Native Keras integration and Hyperband algorithm

## Requirements

The examples have been tested with the following package versions:
- tensorflow==2.15.0
- scikit-learn==1.3.2
- optuna==3.5.0
- keras-tuner==1.4.6

See `requirements.txt` for complete dependencies.

## Troubleshooting

If you encounter CUDA/GPU errors, you can force CPU usage by setting:

```bash
export CUDA_VISIBLE_DEVICES=-1 # On Windows: set CUDA_VISIBLE_DEVICES=-1
```


For any other issues, please check that:
1. Your virtual environment is activated
2. All requirements are installed correctly
3. You're running the scripts from the correct directory
