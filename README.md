# Correlation-Aware Uncertainty Quantification Framework

A comprehensive framework for uncertainty quantification in reinforcement learning that addresses the correlation between epistemic and aleatoric uncertainties. This implementation provides 6 theoretically-grounded uncertainty combination methods as described in the research paper.

## Overview

Traditional uncertainty quantification methods in reinforcement learning assume independence between epistemic (model) and aleatoric (environmental) uncertainties. However, real-world RL scenarios often exhibit complex correlations between these uncertainty types, leading to miscalibrated uncertainty estimates. This framework implements correlation-aware methods that explicitly account for these dependencies.

## Key Features

### Complete Paper Implementation
- **QR-DQN Architecture**: Quantile Regression DQN for aleatoric uncertainty estimation using interquartile range
- **QR-DQN Ensemble**: Deep ensemble of QR-DQN models for epistemic uncertainty through model disagreement
- **6 Uncertainty Combination Methods**: All methods from the paper with exact mathematical formulations
- **Joint Entropy Estimation**: Kernel density estimation for H_joint(s) calculations
- **Comprehensive Evaluation**: Calibration analysis, conformal prediction, and statistical testing

### Architecture

```
correlation_aware_UQ_framework/
├── core/
│   ├── model.py                   # QR-DQN, QR-DQN Ensembles, Action Discretization
│   └── proper_loss_functions.py   # Quantile Regression Loss, Ensemble MSE Loss
├── utils/
│   ├── correlation_analysis.py    # All 6 uncertainty combination methods
│   └── evaluation_metrics.py      # Complete evaluation framework
├── data/
│   └── data_loader.py             # Data loading utilities
├── experiments/
│   └── run_experiments.py         # Main experiment runner
├── src/
│   ├── continuous_uncertainty_framework.py  # Continuous action spaces
│   └── discrete_uncertainty_framework.py    # Discrete action spaces
└── requirements.txt               # All dependencies
```

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

## Evaluation Framework

### Calibration Analysis
- **Expected Calibration Error (ECE)**: Measures calibration quality
- **Maximum Calibration Error (MCE)**: Worst-case calibration error
- **Brier Score**: Probabilistic scoring rule
- **Reliability Diagrams**: Visual calibration assessment

### Conformal Prediction
- **Distribution-free intervals**: Coverage guarantees without distributional assumptions
- **Adaptive sliding window**: Time-varying conformity scores
- **Coverage evaluation**: Statistical validation of prediction intervals

### Statistical Testing
- **Binomial tests**: Coverage rate validation with Wilson confidence intervals
- **Bonferroni correction**: Multiple comparison adjustment
- **Cohen's h**: Effect size estimation for proportions
- **Bootstrap confidence intervals**: Non-parametric uncertainty bounds

## Quick Start

### Installation

```bash
git clone https://github.com/sharyz119/correlation-aware-UQ-framework.git
cd correlation-aware-UQ-framework
pip install -r requirements.txt
```

### Basic Usage

```python
from core.model import create_discrete_uncertainty_models, create_continuous_uncertainty_models
from utils.correlation_analysis import ComprehensiveUncertaintyAnalysis
from utils.evaluation_metrics import ComprehensiveEvaluator

# Create models (user's methodology: QR-DQN + QR-DQN Ensemble)
single_qrdqn, qrdqn_ensemble = create_discrete_uncertainty_models(
    input_size=10, num_actions=4, ensemble_size=3
)

# Extract uncertainties
epistemic = qrdqn_ensemble.get_epistemic_uncertainty(states)  # Ensemble disagreement
aleatoric = single_qrdqn.get_aleatoric_uncertainty(states)    # Quantile spread

# Analyze correlations and combine uncertainties
analyzer = ComprehensiveUncertaintyAnalysis()
results = analyzer.analyze_uncertainty_correlation(epistemic, aleatoric)

# Access all 6 combination methods
methods = results['combinations']
print("Available methods:", list(methods.keys()))
# ['method1_linear_addition', 'method2_rss', 'method3_dcor_entropy', 
#  'method4_nmi_entropy', 'method5_upper_dcor', 'method6_upper_nmi']

# Comprehensive evaluation
evaluator = ComprehensiveEvaluator()
evaluation = evaluator.evaluate_uncertainty_method(
    y_true, y_pred, methods['method5_upper_dcor'], 'Upper dCor'
)
```

### Running Experiments

**Discrete Environments (Atari):**
```bash
cd src
python discrete_uncertainty_framework.py --dataset atari/breakout/expert-v0 --epochs 15
```

**Continuous Environments (MuJoCo):**
```bash
cd src
python continuous_uncertainty_framework.py --env_name halfcheetah --policy_type expert --epochs 15
```

## Expected Results

### Method Performance Ranking
Based on empirical evaluation across multiple environments:

1. **Upper Bound (dCor)** - Best overall performance
2. **Upper Bound (NMI)** - Strong performance with nonlinear correlations
3. **dCor with Entropy Correction** - Good for moderate correlations
4. **NMI with Entropy Correction** - Effective for complex dependencies
5. **Root-Sum-of-Squares** - Reliable baseline
6. **Linear Addition** - Simple but often overestimates

### Correlation Patterns
- **Independent uncertainties**: ρ_dCor ≈ 0.04, ρ_NMI ≈ 0.02
- **Correlated uncertainties**: ρ_dCor ≈ 0.89, ρ_NMI ≈ 0.69
- **Entropy correction**: Visible reduction when correlation is high

## Key Scientific Contributions

1. **First comprehensive correlation analysis** between epistemic and aleatoric uncertainties in RL
2. **Novel entropy-corrected combination methods** using distance correlation and normalized mutual information
3. **Theoretical upper bounds** with interpolation between independence and perfect correlation
4. **Extensive empirical validation** across discrete and continuous control tasks
5. **Complete evaluation framework** with calibration, conformal prediction, and statistical testing

## Experimental Scope

- **Environments**: 6 total (3 Atari + 3 MuJoCo)
- **Policy Types**: Expert, Medium, Simple behavioral policies
- **Sample Size**: ~209,000 total data points across all experiments
- **Evaluation Metrics**: ECE, MCE, Brier Score, Coverage Rate, Statistical Tests

## Advanced Features

### Methodology Consistency
- **QR-DQN for Aleatoric**: Quantile spread (IQR) measures environmental stochasticity
- **QR-DQN Ensemble for Epistemic**: Model disagreement captures parameter uncertainty
- **Architectural Diversity**: Each ensemble member has slight variations for better uncertainty estimation

### Robust Implementation
- **Kernel Density Estimation**: Scott's rule for automatic bandwidth selection
- **Numerical Stability**: Maximum operations prevent negative uncertainty values
- **Error Handling**: Graceful degradation with fallback correlation measures

## Theoretical Background

The framework addresses fundamental questions in uncertainty quantification:

- **When do uncertainties correlate?** High-dimensional, partially observable, sparse-reward domains
- **How to measure correlation?** Distance correlation captures nonlinear dependencies
- **How to combine correlated uncertainties?** Entropy correction prevents double-counting
- **What are the theoretical limits?** Upper bounds provide principled uncertainty estimates

## Contributing

We welcome contributions to improve the framework:

1. **Bug Reports**: Use GitHub issues for bug reports
2. **Feature Requests**: Propose new uncertainty combination methods
3. **Code Contributions**: Follow the existing code style and add tests
4. **Documentation**: Help improve documentation and examples

## Citation

If you use this framework in your research, please cite:

```bibtex
@article{correlation_aware_uq_2024,
  title={Can We Trust Uncertainty? A Correlation-Aware Evaluation Framework for Offline Reinforcement Learning},
  author={[Your Name]},
  journal={[Journal/Conference]},
  year={2024}
}
```

## Contact

For questions, suggestions, or collaborations:
- **GitHub Issues**: [https://github.com/sharyz119/correlation-aware-UQ-framework/issues](https://github.com/sharyz119/correlation-aware-UQ-framework/issues)
- **Email**: [Your Email]

## Related Work

- **Uncertainty Quantification in RL**: Osband et al. (2018), Chua et al. (2018)
- **Distance Correlation**: Székely et al. (2007)
- **Conformal Prediction**: Vovk et al. (2005)
- **Calibration in ML**: Guo et al. (2017)

---

**Framework Status**: ✅ **100% Complete** - All 6 methods implemented and validated 