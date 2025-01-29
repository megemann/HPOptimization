# Machine Learning Examples and Tools

This repository contains various machine learning examples, tools, and implementations focusing on different aspects of ML development and optimization.

## Current Contents

### Hyperparameter Optimization Examples
Located in `code/Corning_HPO_Examples/`, this section demonstrates different approaches to automated hyperparameter optimization using popular frameworks:

- Scikit-learn's RandomizedSearchCV and ParameterSampler
- Optuna
- Keras Tuner

Each implementation shows how to optimize a neural network's hyperparameters using different methodologies. See the directory's [README](code/Corning_HPO_Examples/README.md) for detailed setup and usage instructions.

## Repository Structure

```
.
└── code/
    └── Corning_HPO_Examples/  # Hyperparameter optimization examples
        ├── scikit-learn/      # Scikit-learn based approaches
        ├── optuna_example.py  # Optuna implementation
        └── keras_tuner_example.py  # Keras Tuner implementation
```

## Getting Started

Each subdirectory contains its own README with specific setup and usage instructions. Generally, you'll want to:

1. Create a virtual environment for the specific example you want to run
2. Install the required dependencies from the respective requirements.txt
3. Follow the example-specific README for usage instructions

## Requirements

Different examples may have different requirements. Check the requirements.txt file in each directory for specific package versions needed.

## Contributing

Feel free to contribute by:
1. Opening issues for bugs or enhancement suggestions
2. Creating pull requests for improvements
3. Adding documentation or examples

## License

This project is licensed under the MIT License - see the LICENSE file for details.