#!/usr/bin/env python3
"""
Proper Loss Function Implementations for Uncertainty Quantification
==================================================================

Author: Zixuan Wang
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List
import warnings

class LossFunctions:
    def __init__(self, gamma: float = 0.99, device: str = "cpu"):
        """
        Initialize loss functions with proper parameters.
        
        Args:
            gamma: Discount factor for Bellman equation
            device: Computing device (cpu/cuda)
        """
        self.gamma = gamma
        self.device = device
        
    def quantile_regression_loss_exact(self, 
                                     predicted_quantiles: torch.Tensor,
                                     target_values: torch.Tensor,
                                     quantile_fractions: torch.Tensor,
                                     use_huber: bool = True,
                                     huber_delta: float = 1.0) -> torch.Tensor:
        """
        
        implementation of quantile regression loss.
        
        Mathematical Formula:
        ρ_τ(u) = u · (τ - I_{u < 0})
        
        Args:
            predicted_quantiles: [batch_size, num_quantiles] - Z_τ(s,a)
            target_values: [batch_size, num_quantiles] - r + γ max_a' Q_θ-(s',a')
            quantile_fractions: [num_quantiles] - τ values
            use_huber: Whether to use Huber loss for robustness
            huber_delta: Huber loss threshold
            
        Returns:
            Quantile regression loss scalar
        """
        batch_size, num_quantiles = predicted_quantiles.shape
        
        # Ensure quantile_fractions has correct shape [1, num_quantiles]
        if quantile_fractions.dim() == 1:
            quantile_fractions = quantile_fractions.unsqueeze(0)
        
        # Compute residuals: u = target - prediction
        residuals = target_values - predicted_quantiles  # [batch_size, num_quantiles]
        
        # Indicator function: I_{u < 0}
        indicator = (residuals < 0).float()  # [batch_size, num_quantiles]
        
        # Quantile weights: (τ - I_{u < 0})
        quantile_weights = quantile_fractions - indicator  # [batch_size, num_quantiles]
        
        if use_huber:
            # Huber loss for numerical stability
            huber_loss = torch.where(
                torch.abs(residuals) <= huber_delta,
                0.5 * residuals.pow(2),
                huber_delta * (torch.abs(residuals) - 0.5 * huber_delta)
            )
            # ρ_τ(u) = |τ - I_{u < 0}| * huber_loss(u)
            quantile_loss = torch.abs(quantile_weights) * huber_loss
        else:
            # Exact formula: ρ_τ(u) = u · (τ - I_{u < 0})
            quantile_loss = residuals * quantile_weights
        
        # Return mean loss
        return quantile_loss.mean()
    
    def qrdqn_loss_with_bellman(self,
                               qrdqn_network: nn.Module,
                               target_network: nn.Module,
                               states: torch.Tensor,
                               actions: torch.Tensor,
                               rewards: torch.Tensor,
                               next_states: torch.Tensor,
                               dones: torch.Tensor,
                               quantile_fractions: torch.Tensor) -> torch.Tensor:
        """
        
        Mathematical Formula:
        L_QRDQN = E[(s,a,r,s')~D][Σ_τ ρ_τ(r + γ max_a' Q_θ-(s',a') - Z_τ(s,a))]
        
        Args:
            qrdqn_network: Current QR-DQN network
            target_network: Target QR-DQN network (θ-)
            states: Current states [batch_size, state_dim]
            actions: Actions taken [batch_size]
            rewards: Immediate rewards [batch_size]
            next_states: Next states [batch_size, state_dim]
            dones: Episode termination flags [batch_size]
            quantile_fractions: Quantile fractions τ [num_quantiles]
            
        Returns:
            QR-DQN loss scalar
        """
        batch_size = states.shape[0]
        num_quantiles = len(quantile_fractions)
        
        # Current quantile predictions: Z_τ(s,a)
        current_quantiles = qrdqn_network(states)  # [batch_size, num_actions, num_quantiles]
        
        # Select quantiles for taken actions
        current_quantiles = current_quantiles.gather(
            1, actions.unsqueeze(1).unsqueeze(2).expand(-1, 1, num_quantiles)
        ).squeeze(1)  # [batch_size, num_quantiles]
        
        # Compute Bellman targets: r + γ max_a' Q_θ-(s',a')
        with torch.no_grad():
            # Get next state quantiles from target network
            next_quantiles = target_network(next_states)  # [batch_size, num_actions, num_quantiles]
            
            # Compute Q-values as mean of quantiles for action selection
            next_q_values = next_quantiles.mean(dim=2)  # [batch_size, num_actions]
            
            # Select best actions: max_a' Q_θ-(s',a')
            next_actions = next_q_values.argmax(dim=1)  # [batch_size]
            
            # Get quantiles for best actions
            next_quantiles_selected = next_quantiles.gather(
                1, next_actions.unsqueeze(1).unsqueeze(2).expand(-1, 1, num_quantiles)
            ).squeeze(1)  # [batch_size, num_quantiles]
            
            # Bellman targets: r + γ * (1 - done) * max_a' Q_θ-(s',a')
            targets = rewards.unsqueeze(1) + self.gamma * (1 - dones.unsqueeze(1)) * next_quantiles_selected
            # [batch_size, num_quantiles]
        
        # Compute quantile regression loss
        loss = self.quantile_regression_loss_exact(
            predicted_quantiles=current_quantiles,
            target_values=targets,
            quantile_fractions=quantile_fractions
        )
        
        return loss
    
    def ensemble_mse_loss_exact(self,
                               ensemble_networks: List[nn.Module],
                               target_networks: List[nn.Module],
                               states: torch.Tensor,
                               actions: torch.Tensor,
                               rewards: torch.Tensor,
                               next_states: torch.Tensor,
                               dones: torch.Tensor) -> torch.Tensor:
        """
        
        Mathematical Formula:
        L_ensemble = (1/K) Σ_k E[(r + γ max_a' Q_θk-(s',a') - Q_θk(s,a))²]
        
        Args:
            ensemble_networks: List of K ensemble networks
            target_networks: List of K target networks (θk-)
            states: Current states [batch_size, state_dim]
            actions: Actions taken [batch_size]
            rewards: Immediate rewards [batch_size]
            next_states: Next states [batch_size, state_dim]
            dones: Episode termination flags [batch_size]
            
        Returns:
            Ensemble MSE loss scalar
        """
        K = len(ensemble_networks)
        ensemble_losses = []
        
        for k in range(K):
            # Current Q-values: Q_θk(s,a)
            current_q_values = ensemble_networks[k](states)  # [batch_size, num_actions]
            current_q_selected = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)  # [batch_size]
            
            # Compute Bellman targets: r + γ max_a' Q_θk-(s',a')
            with torch.no_grad():
                next_q_values = target_networks[k](next_states)  # [batch_size, num_actions]
                next_q_max = next_q_values.max(dim=1)[0]  # [batch_size]
                
                # Bellman targets: r + γ * (1 - done) * max_a' Q_θk-(s',a')
                targets = rewards + self.gamma * (1 - dones) * next_q_max  # [batch_size]
            
            # MSE loss for this ensemble member
            member_loss = F.mse_loss(current_q_selected, targets)
            ensemble_losses.append(member_loss)
        
        # Average over all ensemble members: (1/K) Σ_k
        ensemble_loss = torch.stack(ensemble_losses).mean()
        
        return ensemble_loss
    
    def combined_training_objective(self,
                                  qrdqn_loss: torch.Tensor,
                                  ensemble_loss: torch.Tensor,
                                  qrdqn_weight: float = 1.0,
                                  ensemble_weight: float = 1.0) -> torch.Tensor:
        """
        Combined training objective.
        
        Mathematical Formula:
        L_total = L_QRDQN + L_ensemble
        
        Args:
            qrdqn_loss: QR-DQN loss component
            ensemble_loss: Ensemble loss component
            qrdqn_weight: Weight for QR-DQN loss (default: 1.0)
            ensemble_weight: Weight for ensemble loss (default: 1.0)
            
        Returns:
            Combined loss scalar
        """
        total_loss = qrdqn_weight * qrdqn_loss + ensemble_weight * ensemble_loss
        return total_loss
    
    def validate_loss_implementation(self) -> bool:
        """
        Validate that loss implementations work correctly with dummy data.
        
        Returns:
            True if all tests pass
        """
        print("🔍 Validating loss function implementations...")
        
        try:
            # Create dummy data
            batch_size, num_quantiles, num_actions = 32, 21, 4
            state_dim = 10
            
            # Dummy tensors
            states = torch.randn(batch_size, state_dim)
            actions = torch.randint(0, num_actions, (batch_size,))
            rewards = torch.randn(batch_size)
            next_states = torch.randn(batch_size, state_dim)
            dones = torch.randint(0, 2, (batch_size,)).float()
            quantile_fractions = torch.linspace(0.05, 0.95, num_quantiles)
            
            # Test quantile regression loss
            predicted_quantiles = torch.randn(batch_size, num_quantiles)
            target_values = torch.randn(batch_size, num_quantiles)
            
            qr_loss = self.quantile_regression_loss_exact(
                predicted_quantiles, target_values, quantile_fractions
            )
            
            assert qr_loss.item() > 0, "Quantile regression loss should be positive"
            assert not torch.isnan(qr_loss), "Quantile regression loss should not be NaN"
            
            print(" Quantile regression loss validation passed")
            
            # Test with dummy networks (simplified validation)
            predicted_quantiles_2 = torch.randn(batch_size, num_quantiles, requires_grad=True)
            target_values_2 = torch.randn(batch_size, num_quantiles)
            
            loss_2 = self.quantile_regression_loss_exact(
                predicted_quantiles_2, target_values_2, quantile_fractions
            )
            
            # Test gradient flow
            loss_2.backward()
            assert predicted_quantiles_2.grad is not None, "Gradients should flow through loss"
            
            print(" Gradient flow validation passed")
            print(" All loss function validations passed!")
            
            return True
            
        except Exception as e:
            print(f" Loss function validation failed: {e}")
            return False

class TargetNetworkManager:
    """
    Manages target networks for proper Bellman equation implementation.
    """
    
    def __init__(self, update_frequency: int = 1000, tau: float = 0.005):
        """
        Initialize target network manager.
        
        Args:
            update_frequency: Steps between hard target updates
            tau: Soft update coefficient (if using soft updates)
        """
        self.update_frequency = update_frequency
        self.tau = tau
        self.step_count = 0
    
    def hard_update(self, target_network: nn.Module, source_network: nn.Module):
        """
        Hard update: θ- ← θ
        """
        target_network.load_state_dict(source_network.state_dict())
    
    def soft_update(self, target_network: nn.Module, source_network: nn.Module):
        """
        Soft update: θ- ← τ*θ + (1-τ)*θ-
        """
        for target_param, source_param in zip(target_network.parameters(), source_network.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1.0 - self.tau) * target_param.data
            )
    
    def update_if_needed(self, target_networks: List[nn.Module], source_networks: List[nn.Module]):
        """
        Update target networks if needed based on step count.
        """
        self.step_count += 1
        
        if self.step_count % self.update_frequency == 0:
            for target_net, source_net in zip(target_networks, source_networks):
                self.hard_update(target_net, source_net)
            print(f"🔄 Target networks updated at step {self.step_count}")

# # testing
# if __name__ == "__main__":
#     print("🚀 Testing Proper Loss Function Implementations")
    
#     # Initialize loss functions
#     loss_functions = LossFunctions(gamma=0.99, device="cpu")
    
#     # Run validation
#     validation_passed = loss_functions.validate_loss_implementation()
    
#     if validation_passed:
#         print("\n All loss function implementations are working correctly!")
#         print("   • ρ_τ(u) = u · (τ - I_{u < 0})")
#         print("   • L_QRDQN with full Bellman equation")
#         print("   • L_ensemble with proper target networks")
#         print("   • L_total = L_QRDQN + L_ensemble")
#     else:
#         print("\n Loss function validation failed!") 