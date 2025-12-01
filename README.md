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



## Contributing
To ensure higher quality output, this work is still being published in revise, and the code will be kept up-to-date.


## Citation

If you use this framework in your research, please cite:

```bibtex

```

## Contact



## Related Work



