# Benchmark Roadmap

## 1. Smoke Test
- One synthetic function (e.g., Branin)  
- 5–10 min run to verify integration across HPO frameworks  
- Confirm known optimum (~0.3979)  

## 2. Tabular ML Pipelines (Surrogate/Precomputed)
- HPOBench surrogates (XGBoost on Higgs; RF on YearPred2009)  
- YAHPO Gym scenarios (SVM on DNA; MLP on Covertype)  

## 3. Classic UCI / OpenML Tasks
- Adult Income (binary, ~48 K samples)  
- Breast Cancer Wisconsin (binary, ~700 samples)  
- Wine Quality (regression, ~6 K samples)  
- OpenML-100 subset (10–20 diverse tasks)  

## 4. Vision Benchmarks
- MNIST / Fashion-MNIST (28×28 grayscale)  
- CIFAR-10 / CIFAR-100 (32×32 color)  
- SVHN  

## 5. Natural Language Tasks
- IMDb Sentiment (binary classification)  
- AG News (4-class news topic)  
- TREC Question-Type classification  

## 6. Multi-Fidelity & NAS-Style Benchmarks
- NAS-Bench-201 (tabular NAS curves)  
- FCNet (multi-fidelity MLP on UCI)  
- DeepOBS (small CNN training curves)  

## Putting It All Together
- **Smoke Test (Section 1)**: 1 synthetic function, 5–10 min  
- **Surrogates (Section 2)**: 2–3 benchmarks, minutes each  
- **Tabular (Section 3)**: 3–5 datasets, tens of minutes each  
- **Vision (Section 4)**: 2 tasks, ~1 h each  
- **NLP (Section 5)**: 1–2 tasks, ~1 h each  
- **Multi-Fidelity (Section 6)**: 1 benchmark, ~1 h  

_Total: ~15 real benchmarks + 1 quick smoke test_
1. Core Suite in Optuna  
Implement all your paradigms once in Optuna using:  
- `GridSampler`  
- `RandomSampler`  
- `TPESampler`  
- `GPSampler`  
- *(Optional)* `CmaEsSampler`  

That gives you five “pure” algorithmic baselines, all under one API with the exact same `train_evaluate(hparams)` function, budget, and logging.

2. Multi-Fidelity via Keras Tuner Hyperband  
Build a `PyTorchHyperband` (or Keras‐model) wrapper as shown:

```python
from kerastuner.tuners import Hyperband

class PyTorchHyperband(Hyperband):
    def run_trial(self, trial, *args, **kwargs):
        # sample hparams exactly as in Optuna
        # fetch `tuner/epochs` to get this trial’s budget
        # call your shared train_evaluate(hparams, max_epochs)
        # report back val_loss
```

You’ll now have six distinct methods: five Optuna samplers + one Hyperband.

3. Framework Cross-Check on Key Algorithms  
Pick one or two samplers—TPE and/or Random—and re-run them in:  
- HyperOpt (its TPE and Random)  
- Keras Tuner (RandomSearch and/or BayesianOptimization)  
- Ray Tune (OptunaSearch + ASHAScheduler or plain Random by omitting `search_alg`)  

This gives you apples-to-apples insight on:  
> “Is Optuna’s TPE any better than HyperOpt’s TPE? Does RandomSearch behave differently in Keras Tuner vs Optuna vs Ray Tune?”

4. Smoke-Test Everything Once  
For each (framework, algorithm) pairing, do a **5-minute run** on a trivial synthetic (Branin or MNIST-subset) to verify integration. After that, skip repeat smoke tests and pour your remaining hours into the real benchmarks.

5. Suggested 13-Week Timeline  

| Weeks | Tasks                                                      |
|-------|------------------------------------------------------------|
| 1–2   | Shared scaffold: `train_evaluate()`, data loaders, logging |
| 3–6   | Optuna integrations: Grid, Random, TPE, GP, (CMA-ES)       |
| 7–8   | Keras Hyperband integration & smoke tests                  |
| 9     | Cross-framework run: TPE in HyperOpt, Keras BO, Ray Tune   |
| 10    | Cross-framework run: Random in Keras, Ray Tune, HyperOpt   |
| 11–12 | Full runs on all 15+ real benchmarks (tabular, vision, NLP)|
| 13    | Analysis, plots, write-up                                  |

