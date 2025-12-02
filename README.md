# Correlation-Aware Uncertainty Quantification Framework


## Overview

Traditional uncertainty quantification methods in reinforcement learning assume independence between epistemic (model) and aleatoric (environmental) uncertainties. However, real-world RL scenarios often exhibit complex correlations between these uncertainty types, leading to miscalibrated uncertainty estimates. This framework implements correlation-aware methods that explicitly account for these dependencies.

## Key Features

### Complete Paper Implementation
- **QR-DQN Architecture**: Individual Quantile Regression DQN models for aleatoric uncertainty estimation using interquartile range
- **QR-DQN Ensemble**: Deep ensemble of QR-DQN models for epistemic uncertainty through model disagreement
- **6 Uncertainty Combination Methods**: All methods from the paper with exact mathematical formulations
- **Joint Entropy Estimation**: Kernel density estimation for H_joint(s) calculations
- **Comprehensive Evaluation**: Calibration analysis, conformal prediction, and statistical testing

### Methodology

**Aleatoric Uncertainty**: Quantified using the interquartile range (IQR) of quantile predictions from individual QR-DQN models. This captures the inherent stochasticity in the environment.

**Epistemic Uncertainty**: Quantified using the disagreement (variance) between Q-value predictions from an ensemble of QR-DQN models. This captures model uncertainty due to limited data.

**Architecture**: The framework uses an ensemble of QR-DQN models where:
- Each ensemble member is a complete QR-DQN model with architectural diversity
- Aleatoric uncertainty comes from quantile spread within each QR-DQN
- Epistemic uncertainty comes from disagreement between ensemble members


## Uncertainty Combination Methods

### (1) Independence-based Heuristics

**Method 1: Linear Addition (Equation 294)**
```
u_add(s) = u_e(s) + u_a(s)
```

**Method 2: Root-Sum-of-Squares (Equation 300)**
```
u_RSS(s) = √(u_e²(s) + u_a²(s))
```

### (2) Correlation-Aware Combination

**Method 3: Distance Correlation with Entropy Correction (Equation 310)**
```
u_dCor(s) = √(max(0, u_e²(s) + u_a²(s) - ρ_dCor · H_joint(s)))
```

**Method 4: NMI with Entropy Correction (Equation 316)**
```
u_NMI(s) = √(max(0, u_e²(s) + u_a²(s) - ρ_NMI · H_joint(s)))
```

### (3) Upper Bound Formulations

**Method 5: Upper Bound (dCor) (Equation 334)**
```
U_upper_dCor = (1-ρ_dCor)√(U_e² + U_a²) + ρ_dCor·max(U_e, U_a)
```

**Method 6: Upper Bound (NMI) (Equation 338)**
```
U_upper_NMI = (1-ρ_NMI)√(U_e² + U_a²) + ρ_NMI·max(U_e, U_a)
```

## Installation

### Prerequisites

- Python 3.8 or higher
- PyTorch 1.9.0 or higher
- CUDA (optional, for GPU acceleration)

### Optional: Install Minari for Dataset Support

```bash
pip install minari
```

## Quick Start

### Basic Usage: Discrete Action Spaces

```python
from src.discrete_uncertainty_framework import ConsistentDiscreteUncertaintyFramework, ConsistentDiscreteConfig
from pathlib import Path

# Create configuration
config = ConsistentDiscreteConfig(
    num_epochs=15,
    batch_size=128,
    ensemble_size=3,
    num_quantiles=21,
    max_samples=10000,
    results_dir="./results/discrete"
)

# Initialize framework
framework = ConsistentDiscreteUncertaintyFramework(config)

# Run experiment on a dataset
results = framework.run_single_experiment("atari/breakout/expert-v0")
```

### Basic Usage: Continuous Action Spaces

```python
from src.continuous_uncertainty_framework import ConsistentContinuousUncertaintyFramework, ConsistentContinuousConfig

# Create configuration
config = ConsistentContinuousConfig(
    num_epochs=15,
    batch_size=128,
    ensemble_size=3,
    num_quantiles=21,
    max_samples=10000,
    results_dir="./results/continuous"
)

# Initialize framework
framework = ConsistentContinuousUncertaintyFramework(config)

# Run experiment
results = framework.run_single_experiment("halfcheetah", "expert")
```

## Evaluation Metrics

The framework provides comprehensive evaluation metrics:

### Calibration Metrics
- **Expected Calibration Error (ECE)**: Measures how well-calibrated uncertainty estimates are
- **Maximum Calibration Error (MCE)**: Worst-case calibration error
- **Brier Score**: Overall prediction quality metric

### Conformal Prediction
- **Coverage Rate**: Empirical coverage of prediction intervals
- **Interval Width**: Average width of prediction intervals
- **Efficiency**: Coverage per unit interval width

### Statistical Testing
- **Binomial Coverage Tests**: Test if coverage matches target
- **Bonferroni Correction**: Multiple comparison correction
- **Bootstrap Confidence Intervals**: Uncertainty in metrics

### Usage

```python
from utils.evaluation_metrics import ComprehensiveEvaluator

evaluator = ComprehensiveEvaluator(n_bins=10, alpha=0.05)
results = evaluator.evaluate_uncertainty_method(
    y_true, y_pred, uncertainty, method_name="upper_dcor"
)
```



## Citation

If you use this framework in your research, please cite:

```bibtex

```

## Contact

For questions, issues, or collaborations, please open an issue on GitHub.


## Related Work

- [D4RL](https://github.com/rail-berkeley/d4rl): Datasets for Deep Data-Driven Reinforcement Learning
- [Minari](https://github.com/Farama-Foundation/Minari): Standardized offline RL dataset interface
- [QR-DQN](https://arxiv.org/abs/1710.10044): Quantile Regression for Distributional Reinforcement Learning
- 

**Note**: This work is currently under review. The code will be kept up-to-date with the final publication.

