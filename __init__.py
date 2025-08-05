"""
Correlation-Aware Uncertainty Quantification Framework
=====================================================

A comprehensive framework for analyzing and combining epistemic and aleatoric 
uncertainties in reinforcement learning with proper correlation analysis.

Main Components:
- core: Proper loss functions and model architectures
- utils: Correlation analysis and uncertainty combination utilities
- src: Main framework implementations for discrete and continuous environments
- experiments: Experiment management and execution

Author: Uncertainty Correlation Research Team
Date: 2024
"""

__version__ = "1.0.0"
__author__ = "Uncertainty Correlation Research Team"

# Core imports for easy access
from core.proper_loss_functions import ProperLossFunctions, TargetNetworkManager
from core.model_architectures import (
    DiscreteQRDQN, 
    ContinuousQRDQN, 
    DeepEnsemble,
    create_discrete_uncertainty_models,
    create_continuous_uncertainty_models
)
from utils.correlation_analysis import (
    CorrelationAnalyzer,
    UncertaintyCombiner, 
    ComprehensiveUncertaintyAnalysis
)

__all__ = [
    # Core components
    "ProperLossFunctions",
    "TargetNetworkManager",
    "DiscreteQRDQN",
    "ContinuousQRDQN", 
    "DeepEnsemble",
    "create_discrete_uncertainty_models",
    "create_continuous_uncertainty_models",
    
    # Analysis utilities
    "CorrelationAnalyzer",
    "UncertaintyCombiner",
    "ComprehensiveUncertaintyAnalysis",
    
    # Package metadata
    "__version__",
    "__author__"
] 