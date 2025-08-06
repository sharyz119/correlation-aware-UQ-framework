#!/usr/bin/env python3
"""
Model Architectures for Uncertainty Quantification
=================================================

This module implements the network architectures for uncertainty quantification:

1. Quantile Regression DQN (QR-DQN) for aleatoric uncertainty
2. Deep Ensemble of QR-DQN models for epistemic uncertainty
3. Both discrete and continuous action space variants
4. Consistent architectures across environments

Author: Zixuan Wang
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Union
from abc import ABC, abstractmethod

class BaseUncertaintyNetwork(nn.Module, ABC):
    """
    Base class for uncertainty quantification networks.
    """
    
    def __init__(self, input_size: int, output_size: int, hidden_dims: List[int] = None):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_dims = hidden_dims or [128, 128]
    
    @abstractmethod
    def get_uncertainty(self, x: torch.Tensor) -> torch.Tensor:
        """Compute uncertainty measure for input x."""
        pass

class DiscreteQRDQN(BaseUncertaintyNetwork):
    """
    Quantile Regression DQN for discrete action spaces.
    Estimates aleatoric uncertainty through quantile distribution.
    """
    
    def __init__(self, 
                 input_size: int, 
                 num_actions: int, 
                 num_quantiles: int = 21, 
                 hidden_dims: List[int] = None,
                 dropout_rate: float = 0.1,
                 ensemble_id: int = 0):
        super(DiscreteQRDQN, self).__init__(input_size, num_actions, hidden_dims)
        
        self.num_quantiles = num_quantiles
        self.num_actions = num_actions
        self.dropout_rate = dropout_rate
        self.ensemble_id = ensemble_id

        if hidden_dims is None:
            hidden_dims = [128, 128]
        
        layers = []
        prev_dim = input_size
        
        for i, hidden_dim in enumerate(hidden_dims):
            # add architectural diversity for ensemble members
            if ensemble_id > 0:
                varied_hidden = hidden_dim + (ensemble_id - 1) * 16
                varied_hidden = max(64, varied_hidden)
            else:
                varied_hidden = hidden_dim
                
            layers.extend([
                nn.Linear(prev_dim, varied_hidden),
                nn.ReLU(),
                nn.Dropout(dropout_rate + ensemble_id * 0.01)  # Slight dropout variation
            ])
            prev_dim = varied_hidden
        
        # output layer: for each action, output num_quantiles values
        layers.append(nn.Linear(prev_dim, num_actions * num_quantiles))
        
        self.network = nn.Sequential(*layers)
        
        tau_values = [(2*i + 1)/(2*num_quantiles + 1) for i in range(num_quantiles)]
        self.register_buffer('quantile_fractions', torch.tensor(tau_values, dtype=torch.float32))
        
        # initialize weights with ensemble diversity
        self._initialize_weights(seed=ensemble_id * 42)
    
    def _initialize_weights(self, seed: int = 42):
        """Initialize network weights using Xavier initialization with ensemble diversity."""
        torch.manual_seed(seed)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning quantile values.
        
        Args:
            x: Input states [batch_size, input_size]
            
        Returns:
            Quantile values [batch_size, num_actions, num_quantiles]
        """
        batch_size = x.size(0)
        output = self.network(x)  # [batch_size, num_actions * num_quantiles]
        
        # reshape to [batch_size, num_actions, num_quantiles]
        return output.view(batch_size, self.num_actions, self.num_quantiles)
    
    def get_q_values(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get Q-values as mean of quantiles.
        
        Args:
            x: Input states [batch_size, input_size]
            
        Returns:
            Q-values [batch_size, num_actions]
        """
        quantiles = self.forward(x)
        return quantiles.mean(dim=-1)
    
    def get_aleatoric_uncertainty(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute aleatoric uncertainty from quantile spread within this QR-DQN.
        
        Args:
            x: Input states [batch_size, input_size]
            
        Returns:
            Aleatoric uncertainty [batch_size]
        """
        with torch.no_grad():
            quantiles = self.forward(x)  # [batch_size, num_actions, num_quantiles]
            
            # use interquartile range (IQR) as uncertainty measure
            q75 = torch.quantile(quantiles, 0.75, dim=-1)  # [batch_size, num_actions]
            q25 = torch.quantile(quantiles, 0.25, dim=-1)  # [batch_size, num_actions]
            iqr = q75 - q25  # [batch_size, num_actions]
            
            # return mean IQR across actions
            return iqr.mean(dim=-1)  # [batch_size]
    
    def get_uncertainty(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for get_aleatoric_uncertainty for compatibility."""
        return self.get_aleatoric_uncertainty(x)

class ContinuousQRDQN(BaseUncertaintyNetwork):
    """
    Quantile Regression DQN adapted for continuous action spaces.
    Uses action discretization for compatibility.
    """
    
    def __init__(self, 
                 state_dim: int, 
                 action_dim: int,
                 action_discretization_bins: int = 7,
                 num_quantiles: int = 21, 
                 hidden_dims: List[int] = None,
                 dropout_rate: float = 0.1,
                 ensemble_id: int = 0):
        super(ContinuousQRDQN, self).__init__(state_dim, action_discretization_bins**action_dim, hidden_dims)
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_discretization_bins = action_discretization_bins
        self.num_quantiles = num_quantiles
        self.dropout_rate = dropout_rate
        self.ensemble_id = ensemble_id
        
        # total number of discrete actions (bins^action_dim)
        self.total_discrete_actions = action_discretization_bins ** action_dim
        

        if hidden_dims is None:
            hidden_dims = [256, 256]
        
        # build the network layers with ensemble diversity
        layers = []
        prev_dim = state_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            # add architectural diversity for ensemble members
            if ensemble_id > 0:
                varied_hidden = hidden_dim + (ensemble_id - 1) * 16
                varied_hidden = max(64, varied_hidden)
            else:
                varied_hidden = hidden_dim
                
            layers.extend([
                nn.Linear(prev_dim, varied_hidden),
                nn.ReLU(),
                nn.Dropout(dropout_rate + ensemble_id * 0.01)
            ])
            prev_dim = varied_hidden
        
        # Output layer: for each discrete action, output num_quantiles values
        layers.append(nn.Linear(prev_dim, self.total_discrete_actions * num_quantiles))
        
        self.network = nn.Sequential(*layers)
        tau_values = [(2*i + 1)/(2*num_quantiles + 1) for i in range(num_quantiles)]
        self.register_buffer('quantile_fractions', torch.tensor(tau_values, dtype=torch.float32))
        
        # initialize weights with ensemble diversity
        self._initialize_weights(seed=ensemble_id * 42)
    
    def _initialize_weights(self, seed: int = 42):
        """Initialize weights with ensemble diversity."""
        torch.manual_seed(seed)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning quantiles for all discrete actions.
        
        Args:
            state: Input states [batch_size, state_dim]
            
        Returns:
            Quantile values [batch_size, total_discrete_actions, num_quantiles]
        """
        batch_size = state.size(0)
        output = self.network(state)
        return output.view(batch_size, self.total_discrete_actions, self.num_quantiles)
    
    def get_q_values(self, state: torch.Tensor) -> torch.Tensor:
        """
        Get Q-values as mean of quantiles.
        
        Args:
            state: Input states [batch_size, state_dim]
            
        Returns:
            Q-values [batch_size, total_discrete_actions]
        """
        quantiles = self.forward(state)
        return quantiles.mean(dim=-1)
    
    def get_aleatoric_uncertainty(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute aleatoric uncertainty from quantile spread within this QR-DQN.
        
        Args:
            state: Input states [batch_size, state_dim]
            
        Returns:
            Aleatoric uncertainty [batch_size]
        """
        with torch.no_grad():
            quantiles = self.forward(state)
            
            # use interquartile range as uncertainty measure
            q75 = torch.quantile(quantiles, 0.75, dim=-1)
            q25 = torch.quantile(quantiles, 0.25, dim=-1)
            iqr = q75 - q25
            
            # return mean IQR across all discrete actions
            return iqr.mean(dim=-1)
    
    def get_uncertainty(self, state: torch.Tensor) -> torch.Tensor:
        """Alias for get_aleatoric_uncertainty for compatibility."""
        return self.get_aleatoric_uncertainty(state)

class QRDQNEnsemble(nn.Module):
    """
    Deep ensemble of QR-DQN models for epistemic uncertainty quantification.
    
    - Aleatoric uncertainty: quantile spread within each QR-DQN
    - Epistemic uncertainty: disagreement between QR-DQN ensemble members
    """
    
    def __init__(self, 
                 input_size: int, 
                 num_actions: int, 
                 ensemble_size: int = 3,
                 num_quantiles: int = 21,
                 hidden_dims: List[int] = None,
                 dropout_rate: float = 0.1,
                 is_continuous: bool = False,
                 action_dim: int = None,
                 action_discretization_bins: int = 7):
        """
        Initialize ensemble of QR-DQN models.
        
        Args:
            input_size: Input dimension (state_dim for continuous, input_size for discrete)
            num_actions: Number of actions (or total discrete actions for continuous)
            ensemble_size: Number of QR-DQN ensemble members
            num_quantiles: Number of quantiles per QR-DQN
            hidden_dims: Hidden layer dimensions
            dropout_rate: Dropout rate
            is_continuous: Whether this is for continuous action spaces
            action_dim: Action dimension (for continuous only)
            action_discretization_bins: Discretization bins (for continuous only)
        """
        super().__init__()
        self.input_size = input_size
        self.num_actions = num_actions
        self.ensemble_size = ensemble_size
        self.num_quantiles = num_quantiles
        self.hidden_dims = hidden_dims or [128, 128]
        self.is_continuous = is_continuous
        
        # create ensemble of QR-DQN models
        self.qrdqn_ensemble = nn.ModuleList()
        
        for i in range(ensemble_size):
            if is_continuous:
                qrdqn = ContinuousQRDQN(
                    state_dim=input_size,
                    action_dim=action_dim,
                    action_discretization_bins=action_discretization_bins,
                    num_quantiles=num_quantiles,
                    hidden_dims=hidden_dims,
                    dropout_rate=dropout_rate,
                    ensemble_id=i
                )
            else:
                qrdqn = DiscreteQRDQN(
                    input_size=input_size,
                    num_actions=num_actions,
                    num_quantiles=num_quantiles,
                    hidden_dims=hidden_dims,
                    dropout_rate=dropout_rate,
                    ensemble_id=i
                )
            self.qrdqn_ensemble.append(qrdqn)
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass through all QR-DQN ensemble members.
        
        Args:
            x: Input tensor [batch_size, input_size]
            
        Returns:
            List of quantile outputs from each QR-DQN member
        """
        outputs = []
        for qrdqn in self.qrdqn_ensemble:
            outputs.append(qrdqn(x))
        return outputs
    
    def get_mean_quantiles(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get mean quantiles across ensemble.
        
        Args:
            x: Input tensor [batch_size, input_size]
            
        Returns:
            Mean quantiles [batch_size, num_actions, num_quantiles]
        """
        quantile_outputs = self.forward(x)
        stacked = torch.stack(quantile_outputs, dim=0)  # [ensemble_size, batch_size, num_actions, num_quantiles]
        return stacked.mean(dim=0)
    
    def get_mean_q_values(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get mean Q-values across ensemble.
        
        Args:
            x: Input tensor [batch_size, input_size]
            
        Returns:
            Mean Q-values [batch_size, num_actions]
        """
        mean_quantiles = self.get_mean_quantiles(x)
        return mean_quantiles.mean(dim=-1)  # Average over quantiles
    
    def get_aleatoric_uncertainty(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute aleatoric uncertainty from average quantile spread across ensemble.
        
        This represents the inherent stochasticity in the environment.
        
        Args:
            x: Input tensor [batch_size, input_size]
            
        Returns:
            Aleatoric uncertainty [batch_size]
        """
        with torch.no_grad():
            # get aleatoric uncertainty from each QR-DQN and average
            aleatoric_uncertainties = []
            for qrdqn in self.qrdqn_ensemble:
                aleatoric_uncertainties.append(qrdqn.get_aleatoric_uncertainty(x))
            
            # Average aleatoric uncertainty across ensemble members
            stacked = torch.stack(aleatoric_uncertainties, dim=0)  # [ensemble_size, batch_size]
            return stacked.mean(dim=0)  # [batch_size]
    
    def get_epistemic_uncertainty(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute epistemic uncertainty from disagreement between QR-DQN ensemble members.
        
        This represents model uncertainty due to limited data.
        
        Args:
            x: Input tensor [batch_size, input_size]
            
        Returns:
            Epistemic uncertainty [batch_size]
        """
        with torch.no_grad():
            # get Q-values from each ensemble member
            q_values_list = []
            for qrdqn in self.qrdqn_ensemble:
                q_values_list.append(qrdqn.get_q_values(x))
            
            # stack and compute variance across ensemble members
            stacked_q_values = torch.stack(q_values_list, dim=0)  # [ensemble_size, batch_size, num_actions]
            
            # use variance across ensemble as epistemic uncertainty
            variance = torch.var(stacked_q_values, dim=0)  # [batch_size, num_actions]
            
            # return mean variance across actions
            return variance.mean(dim=-1)  # [batch_size]
    
    def get_uncertainty(self, x: torch.Tensor) -> torch.Tensor:
        """
        Default uncertainty measure (epistemic uncertainty).
        
        Args:
            x: Input tensor [batch_size, input_size]
            
        Returns:
            Epistemic uncertainty [batch_size]
        """
        return self.get_epistemic_uncertainty(x)
    
    def get_both_uncertainties(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get both aleatoric and epistemic uncertainties.
        
        Args:
            x: Input tensor [batch_size, input_size]
            
        Returns:
            Tuple of (epistemic_uncertainty, aleatoric_uncertainty)
        """
        epistemic = self.get_epistemic_uncertainty(x)
        aleatoric = self.get_aleatoric_uncertainty(x)
        return epistemic, aleatoric
    
    def get_ensemble_statistics(self, x: torch.Tensor) -> dict:
        """
        Get comprehensive ensemble statistics.
        
        Args:
            x: Input tensor [batch_size, input_size]
            
        Returns:
            Dictionary with ensemble statistics
        """
        with torch.no_grad():
            # get outputs from all ensemble members
            quantile_outputs = self.forward(x)
            q_values_list = [qrdqn.get_q_values(x) for qrdqn in self.qrdqn_ensemble]
            
            # stack for statistics
            stacked_quantiles = torch.stack(quantile_outputs, dim=0)  # [ensemble_size, batch_size, num_actions, num_quantiles]
            stacked_q_values = torch.stack(q_values_list, dim=0)  # [ensemble_size, batch_size, num_actions]
            
            return {
                'mean_quantiles': stacked_quantiles.mean(dim=0),
                'std_quantiles': stacked_quantiles.std(dim=0),
                'mean_q_values': stacked_q_values.mean(dim=0),
                'std_q_values': stacked_q_values.std(dim=0),
                'epistemic_uncertainty': self.get_epistemic_uncertainty(x),
                'aleatoric_uncertainty': self.get_aleatoric_uncertainty(x),
                'individual_quantiles': quantile_outputs,
                'individual_q_values': q_values_list
            }

class ActionDiscretizer:
    """
    Utility class for discretizing continuous actions for QR-DQN compatibility.
    """
    
    def __init__(self, 
                 action_dim: int, 
                 bins: int = 7, 
                 action_bounds: Tuple[float, float] = (-1.0, 1.0)):
        """
        Initialize action discretizer.
        
        Args:
            action_dim: Continuous action dimension
            bins: Number of bins per action dimension
            action_bounds: (min, max) bounds for actions
        """
        self.action_dim = action_dim
        self.bins = bins
        self.action_bounds = action_bounds
        
        # Create discretization bins for each action dimension
        self.bin_edges = np.linspace(action_bounds[0], action_bounds[1], bins + 1)
        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2
        
        # Total number of discrete actions
        self.total_discrete_actions = bins ** action_dim
    
    def discretize_action(self, continuous_action: np.ndarray) -> int:
        """
        Convert continuous action to discrete action index.
        
        Args:
            continuous_action: Continuous action vector [action_dim]
            
        Returns:
            Discrete action index
        """
        # clip actions to bounds
        clipped_action = np.clip(continuous_action, self.action_bounds[0], self.action_bounds[1])
        
        # find bin index for each dimension
        bin_indices = np.digitize(clipped_action, self.bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, self.bins - 1)
        
        # convert multi-dimensional bin indices to single discrete action
        discrete_action = 0
        for i, bin_idx in enumerate(bin_indices):
            discrete_action += bin_idx * (self.bins ** i)
            
        return discrete_action
    
    def batch_discretize(self, continuous_actions: np.ndarray) -> np.ndarray:
        """
        Batch discretize continuous actions.
        
        Args:
            continuous_actions: Continuous actions [batch_size, action_dim]
            
        Returns:
            Discrete action indices [batch_size]
        """
        return np.array([self.discretize_action(action) for action in continuous_actions])
    
    def undiscretize_action(self, discrete_action: int) -> np.ndarray:
        """
        Convert discrete action index back to continuous action.
        
        Args:
            discrete_action: Discrete action index
            
        Returns:
            Continuous action vector [action_dim]
        """
        continuous_action = np.zeros(self.action_dim)
        remaining = discrete_action
        
        for i in range(self.action_dim):
            bin_idx = remaining % self.bins
            continuous_action[i] = self.bin_centers[bin_idx]
            remaining //= self.bins
        
        return continuous_action

# factory functions for easy model creation
def create_discrete_uncertainty_models(input_size: int, 
                                     num_actions: int,
                                     ensemble_size: int = 3,
                                     num_quantiles: int = 21,
                                     hidden_dims: List[int] = None) -> Tuple[DiscreteQRDQN, QRDQNEnsemble]:
    """
    Create single QR-DQN and QR-DQN ensemble for discrete action spaces.
    
    - Single QR-DQN for aleatoric uncertainty estimation
    - QR-DQN ensemble for epistemic uncertainty estimation
    
    Args:
        input_size: State dimension
        num_actions: Number of discrete actions
        ensemble_size: Number of ensemble members
        num_quantiles: Number of quantiles for QR-DQN
        hidden_dims: Hidden layer dimensions
        
    Returns:
        Tuple of (Single QR-DQN, QR-DQN Ensemble)
    """
    # single QR-DQN for aleatoric uncertainty
    single_qrdqn = DiscreteQRDQN(
        input_size=input_size,
        num_actions=num_actions,
        num_quantiles=num_quantiles,
        hidden_dims=hidden_dims,
        ensemble_id=0  # Base model
    )
    
    # QR-DQN ensemble for epistemic uncertainty
    qrdqn_ensemble = QRDQNEnsemble(
        input_size=input_size,
        num_actions=num_actions,
        ensemble_size=ensemble_size,
        num_quantiles=num_quantiles,
        hidden_dims=hidden_dims,
        is_continuous=False
    )
    
    return single_qrdqn, qrdqn_ensemble

def create_continuous_uncertainty_models(state_dim: int,
                                       action_dim: int,
                                       action_discretization_bins: int = 7,
                                       ensemble_size: int = 3,
                                       num_quantiles: int = 21,
                                       hidden_dims: List[int] = None) -> Tuple[ContinuousQRDQN, QRDQNEnsemble, ActionDiscretizer]:
    """
    Create single QR-DQN and QR-DQN ensemble for continuous action spaces.

    - Single QR-DQN for aleatoric uncertainty estimation
    - QR-DQN ensemble for epistemic uncertainty estimation
    
    Args:
        state_dim: State space dimension
        action_dim: Continuous action dimension
        action_discretization_bins: Number of bins per action dimension
        ensemble_size: Number of ensemble members
        num_quantiles: Number of quantiles for QR-DQN
        hidden_dims: Hidden layer dimensions
        
    Returns:
        Tuple of (Single QR-DQN, QR-DQN Ensemble, Action Discretizer)
    """
    # single QR-DQN for aleatoric uncertainty
    single_qrdqn = ContinuousQRDQN(
        state_dim=state_dim,
        action_dim=action_dim,
        action_discretization_bins=action_discretization_bins,
        num_quantiles=num_quantiles,
        hidden_dims=hidden_dims,
        ensemble_id=0  # Base model
    )
    
    # QR-DQN ensemble for epistemic uncertainty
    qrdqn_ensemble = QRDQNEnsemble(
        input_size=state_dim,
        num_actions=action_discretization_bins ** action_dim,
        ensemble_size=ensemble_size,
        num_quantiles=num_quantiles,
        hidden_dims=hidden_dims,
        is_continuous=True,
        action_dim=action_dim,
        action_discretization_bins=action_discretization_bins
    )
    
    # action discretizer
    action_discretizer = ActionDiscretizer(
        action_dim=action_dim,
        bins=action_discretization_bins
    )
    
    return single_qrdqn, qrdqn_ensemble, action_discretizer

# # testing
# if __name__ == "__main__":
#     print("Testing Model Architectures")
#     print("=" * 50)
    
#     # Test discrete models
#     print("\nTesting Discrete Models:")
#     input_size, num_actions = 10, 4
#     single_qrdqn, qrdqn_ensemble = create_discrete_uncertainty_models(
#         input_size=input_size,
#         num_actions=num_actions,
#         ensemble_size=3
#     )
    
#     # Test forward pass
#     batch_size = 32
#     x = torch.randn(batch_size, input_size)
    
#     # Single QR-DQN (for aleatoric uncertainty)
#     quantiles = single_qrdqn(x)
#     aleatoric_uncertainty = single_qrdqn.get_aleatoric_uncertainty(x)
    
#     print(f"   Single QR-DQN quantiles shape: {quantiles.shape}")
#     print(f"   Aleatoric uncertainty shape: {aleatoric_uncertainty.shape}")
    
#     # QR-DQN Ensemble (for epistemic uncertainty)
#     epistemic_uncertainty = qrdqn_ensemble.get_epistemic_uncertainty(x)
#     ensemble_aleatoric = qrdqn_ensemble.get_aleatoric_uncertainty(x)
#     mean_q_values = qrdqn_ensemble.get_mean_q_values(x)
    
#     print(f"   Ensemble epistemic uncertainty shape: {epistemic_uncertainty.shape}")
#     print(f"   Ensemble aleatoric uncertainty shape: {ensemble_aleatoric.shape}")
#     print(f"   Ensemble mean Q-values shape: {mean_q_values.shape}")
    
#     # Test continuous models
#     print("\nTesting Continuous Models:")
#     state_dim, action_dim = 17, 6
#     single_qrdqn_cont, qrdqn_ensemble_cont, discretizer = create_continuous_uncertainty_models(
#         state_dim=state_dim,
#         action_dim=action_dim,
#         ensemble_size=3
#     )
    
#     # Test forward pass
#     states = torch.randn(batch_size, state_dim)
    
#     # Single QR-DQN (for aleatoric uncertainty)
#     quantiles_cont = single_qrdqn_cont(states)
#     aleatoric_uncertainty_cont = single_qrdqn_cont.get_aleatoric_uncertainty(states)
    
#     print(f"   Continuous QR-DQN quantiles shape: {quantiles_cont.shape}")
#     print(f"   Continuous aleatoric uncertainty shape: {aleatoric_uncertainty_cont.shape}")
    
#     # QR-DQN Ensemble (for epistemic uncertainty)
#     epistemic_uncertainty_cont = qrdqn_ensemble_cont.get_epistemic_uncertainty(states)
#     ensemble_aleatoric_cont = qrdqn_ensemble_cont.get_aleatoric_uncertainty(states)
    
#     print(f"   Continuous ensemble epistemic uncertainty shape: {epistemic_uncertainty_cont.shape}")
#     print(f"   Continuous ensemble aleatoric uncertainty shape: {ensemble_aleatoric_cont.shape}")
    
#     # Test both uncertainties together
#     epistemic, aleatoric = qrdqn_ensemble_cont.get_both_uncertainties(states)
#     print(f"   Both uncertainties - Epistemic: {epistemic.shape}, Aleatoric: {aleatoric.shape}")
    
#     # Test action discretizer
#     continuous_actions = np.random.uniform(-1, 1, (10, action_dim))
#     discrete_actions = discretizer.batch_discretize(continuous_actions)
    
#     print(f"   Action discretizer: {action_dim}D -> {discretizer.total_discrete_actions} discrete actions")
#     print(f"   Sample discrete actions: {discrete_actions[:5]}")
    
#     print("\nMethodology Verification:")
#     print("- Aleatoric uncertainty: QR-DQN quantile spread")
#     print("- Epistemic uncertainty: QR-DQN ensemble disagreement") 
#     print("- Architecture: Ensemble of QR-DQN models")
#     print("\nAll model architectures tested successfully!") 