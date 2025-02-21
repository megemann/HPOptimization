# Automated Hyperparameter Optimization Guide & Examples

A comprehensive guide and implementation repository for automated hyperparameter optimization research. This repository combines theoretical understanding with practical implementations.

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

## Repository Structure

```
.
├── docs/                      # Technical documentation and research
│   └── HP_Optimization_Manual.pdf             # Comprehensive HPO guide
└── code/
    └── HPO_Manual_Examples/ # Implementation examples
        ├── base_model.py # Base model for all examples
        ├── scikit-learn/     # Scikit-learn based approaches
        ├── optuna_example.py # Optuna implementation
        └── keras_tuner_example.py # Keras Tuner implementation
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
