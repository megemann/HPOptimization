# Automated Hyperparameter Optimization Guide & Examples

A comprehensive guide and implementation repository for automated hyperparameter optimization research. This repository combines theoretical understanding with practical implementations, including a custom Hyperband sampler for Optuna.

## Repository Contents

### Documentation & Research
Located in `/docs`, this technical manual provides:
- Detailed explanations of hyperparameter optimization techniques
- Comparative analysis of different frameworks
- Best practices and implementation strategies
- Common pitfalls and solutions
- Advanced usage patterns and considerations
- Analysis of search algorithms and their impact on HPO

1. **HP_Optimization_Manual**: This was developed during in Summer 2024, when researching HP Optimization frameworks
2.  **Independent_Study_Proposal**: Document developed for an independent study - now planning to start in fall 2025

### Implementation Examples
Located in `code/HPO_Manual_Examples/`, this section provides working implementations using popular frameworks:
- Scikit-learn's RandomizedSearchCV and ParameterSampler
- Optuna
- Keras Tuner

Each implementation demonstrates practical application of concepts discussed in the manual. See the directory's [README](code/HPO_Manual_Examples/README.md) for setup and usage instructions.

### Hyperband Sampler Package
Located in `hyperband_sampler/` (as a git submodule), this package provides:
- Custom Hyperband sampler implementation for Optuna
- Multi-objective optimization support
- Parallel execution capabilities
- Enhanced timeout handling

## Installation

### Installing the Hyperband Sampler Package

You can install this repository and the hyperband sampler package in several ways:

#### Option 1: Install from source (recommended for development)
```bash
# Clone the repository with submodules
git clone --recursive https://github.com/megemann/HPOptimization.git
cd HPOptimization

# Install in development mode
pip install -e .
```

#### Option 2: Install just the hyperband sampler
```bash
# Clone and install only the hyperband sampler
git clone https://github.com/megemann/Hyperband_sampler.git
cd Hyperband_sampler
pip install -e .
```

#### Option 3: Install with optional dependencies
```bash
# Install with PyTorch support
pip install -e ".[torch]"

# Install with machine learning utilities
pip install -e ".[ml]"

# Install with all optional dependencies
pip install -e ".[all]"

# Install with development tools
pip install -e ".[dev]"
```

### Usage Example
```python
from hyperband_sampler import HyperbandSampler, HyperbandStudy
import optuna

# Create a study with the Hyperband sampler
study = optuna.create_study(
    sampler=HyperbandSampler(
        min_resource=1,
        max_resource=100,
        reduction_factor=3
    )
)

# Or use the HyperbandStudy wrapper for multiple iterations
hyperband_study = HyperbandStudy(
    min_resource=1,
    max_resource=100,
    reduction_factor=3,
    hyperband_iterations=5
)

# Run optimization
hyperband_study.optimize(objective_function, n_jobs=4)
```

## Repository Structure

```
.
├── docs/                      # Technical documentation and research
│   └── HP_Optimization_Manual.pdf             # Comprehensive HPO guide
├── hyperband_sampler/         # Hyperband sampler package (git submodule)
│   ├── __init__.py           # Package initialization
│   ├── hyperband_sampler.py  # Main sampler implementation
│   ├── hyperband_study.py    # Study wrapper with multi-iteration support
│   └── requirements.txt      # Package dependencies
├── code/
│   └── HPO_Manual_Examples/  # Implementation examples
│       ├── base_model.py     # Base model for all examples
│       ├── scikit-learn/     # Scikit-learn based approaches
│       ├── optuna_example.py # Optuna implementation
│       └── keras_tuner_example.py # Keras Tuner implementation
├── pyproject.toml            # Modern Python package configuration
├── setup.py                  # Backward compatibility setup
└── MANIFEST.in              # Package distribution files
```

## Getting Started

### Reading the Manual
Start with the manual in `/docs` to understand:
1. Core concepts of hyperparameter optimization
2. Different approaches and their trade-offs
3. Framework-specific considerations
4. Implementation strategies

### Running Examples
Each implementation directory contains:
1. Specific setup instructions
2. Requirements.txt for dependencies
3. Usage examples and explanations

## Requirements

Different examples have different requirements. Check individual requirements.txt files for specific package versions needed.

## Contributing

If you have any suggestions or improvements, please feel free to open an issue or submit a pull request. If there is enough demand, I may implement the advanced solutions later.
Additionally, you can contact me at ajfairbanks2005@gmail.com.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
