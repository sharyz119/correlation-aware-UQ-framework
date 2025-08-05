#!/usr/bin/env python3
"""
Continuous Environment Uncertainty Correlation Study
=========================================================================

Complete implementation of uncertainty correlation analysis for continuous action spaces,
consistent with discrete implementation except for necessary environment differences.


Author: Zixuan Wang
"""

import os
import sys
import json
import warnings
import gc
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass
import time
import traceback

# Set matplotlib backend
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import pdist, squareform
import pandas as pd
import h5py
import pickle

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

from loss_functions import LossFunctions, TargetNetworkManager

@dataclass
class ConsistentContinuousConfig:
    """Consistent configuration for continuous uncertainty experiments."""
    # Training parameters - CONSISTENT WITH DISCRETE
    num_epochs: int = 15
    batch_size: int = 128
    learning_rate: float = 1e-4
    max_samples: int = 10000
    
    # Model parameters - CONSISTENT WITH DISCRETE
    ensemble_size: int = 3
    num_quantiles: int = 21
    hidden_dims: List[int] = None  # Will be set to [128, 128]
    
    # Action discretization for QRDQN - CONTINUOUS SPECIFIC
    action_discretization_bins: int = 7
    use_action_discretization: bool = True
    
    # Analysis parameters - CONSISTENT WITH DISCRETE
    correlation_threshold: float = 0.3
    nmi_cap: float = 0.8
    
    # Dataset parameters
    dataset_path: str = "/var/scratch/zwa212/.minari/datasets/mujoco"
    environments: List[str] = None
    policies: List[str] = None
    
    # Output parameters
    results_dir: str = "/var/scratch/zwa212/UQ_continuous_consistent_results"
    save_plots: bool = True
    save_models: bool = False  # Consistent with discrete
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [128, 128]  # Consistent with discrete
        if self.environments is None:
            self.environments = ["halfcheetah", "hopper", "walker2d"]
        if self.policies is None:
            self.policies = ["expert", "medium", "simple"]

class ActionDiscretizer:
    """Discretizes continuous actions for QRDQN compatibility - CONTINUOUS SPECIFIC."""
    
    def __init__(self, action_dim: int, bins: int = 7, action_bounds: Tuple[float, float] = (-1.0, 1.0)):
        self.action_dim = action_dim
        self.bins = bins
        self.action_bounds = action_bounds
        
        # Create discretization bins for each action dimension
        self.bin_edges = np.linspace(action_bounds[0], action_bounds[1], bins + 1)
        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2
        
        # Total number of discrete actions (bins^action_dim)
        self.total_discrete_actions = bins ** action_dim
        
    def discretize_action(self, continuous_action: np.ndarray) -> np.ndarray:
        """Convert continuous action to discrete action index."""
        # Clip actions to bounds
        clipped_action = np.clip(continuous_action, self.action_bounds[0], self.action_bounds[1])
        
        # Find bin index for each dimension
        bin_indices = np.digitize(clipped_action, self.bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, self.bins - 1)
        
        # Convert multi-dimensional bin indices to single discrete action
        discrete_action = 0
        for i, bin_idx in enumerate(bin_indices):
            discrete_action += bin_idx * (self.bins ** i)
            
        return discrete_action
    
    def batch_discretize(self, continuous_actions: np.ndarray) -> np.ndarray:
        """Batch discretize continuous actions."""
        return np.array([self.discretize_action(action) for action in continuous_actions])

class ContinuousQRDQN(nn.Module):
    """Quantile Regression DQN for continuous action spaces - CONSISTENT ARCHITECTURE."""
    
    def __init__(self, state_dim: int, action_dim: int, num_quantiles: int = 21, hidden_dims: List[int] = None):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim  # the total discrete actions
        self.num_quantiles = num_quantiles
        
        if hidden_dims is None:
            hidden_dims = [128, 128]
        
        # Network architecture - CONSISTENT WITH DISCRETE
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)  # Consistent dropout rate
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, action_dim * num_quantiles))
        
        self.network = nn.Sequential(*layers)
        
        # Quantile fractions - CONSISTENT WITH DISCRETE
        self.register_buffer('quantile_fractions', 
                           torch.linspace(0.1, 0.9, num_quantiles))
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass returning quantiles for all actions."""
        batch_size = state.size(0)
        output = self.network(state)
        return output.view(batch_size, self.action_dim, self.num_quantiles)
    
    def get_aleatoric_uncertainty(self, state: torch.Tensor) -> torch.Tensor:
        """Compute aleatoric uncertainty from quantile spread - CONSISTENT METHOD."""
        with torch.no_grad():
            quantiles = self.forward(state)
            # Use interquartile range as uncertainty measure - CONSISTENT
            q75 = torch.quantile(quantiles, 0.75, dim=-1)
            q25 = torch.quantile(quantiles, 0.25, dim=-1)
            uncertainty = q75 - q25
            return uncertainty.mean(dim=-1)  # Average over actions

class ContinuousEnsemble(nn.Module):
    """Deep ensemble for epistemic uncertainty - CONSISTENT ARCHITECTURE."""
    
    def __init__(self, state_dim: int, action_dim: int, ensemble_size: int = 3, hidden_dims: List[int] = None):
        super().__init__()
        self.ensemble_size = ensemble_size
        self.action_dim = action_dim
        
        if hidden_dims is None:
            hidden_dims = [128, 128]
        
        # Create ensemble members - CONSISTENT WITH DISCRETE
        self.ensemble = nn.ModuleList()
        for i in range(ensemble_size):
            layers = []
            prev_dim = state_dim
            
            for j, hidden_dim in enumerate(hidden_dims):
                # Slight architectural diversity for ensemble members
                current_hidden = hidden_dim + (i - 1) * 16  # Small variation
                current_hidden = max(64, current_hidden)
                
                layers.extend([
                    nn.Linear(prev_dim, current_hidden),
                    nn.ReLU(),
                    nn.Dropout(0.1 + i * 0.01)  # Slight dropout variation
                ])
                prev_dim = current_hidden
            
            # Output layer for continuous actions
            layers.append(nn.Linear(prev_dim, action_dim))
            
            member = nn.Sequential(*layers)
            self.ensemble.append(member)
    
    def forward(self, state: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass through all ensemble members."""
        outputs = []
        for member in self.ensemble:
            outputs.append(member(state))
        return outputs
    
    def get_epistemic_uncertainty(self, state: torch.Tensor) -> torch.Tensor:
        """Compute epistemic uncertainty from ensemble disagreement - CONSISTENT METHOD."""
        with torch.no_grad():
            outputs = self.forward(state)
            stacked = torch.stack(outputs, dim=0)  # [ensemble_size, batch_size, action_dim]
            
            # Use variance across ensemble as epistemic uncertainty
            uncertainty = torch.var(stacked, dim=0)  # [batch_size, action_dim]
            return uncertainty.mean(dim=-1)  # Average over actions

class ConsistentContinuousUncertaintyFramework:
    """Main framework for continuous uncertainty correlation analysis - CONSISTENT VERSION."""
    
    def __init__(self, config: ConsistentContinuousConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Create results directory
        Path(config.results_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize data storage - CONSISTENT WITH DISCRETE
        self.reset_storage()
        
        print(f"🌊 Consistent Continuous Uncertainty Framework initialized")
        print(f"   Device: {self.device}")
        print(f"   Results: {config.results_dir}")
        print(f"   Architecture: {config.hidden_dims}")
        print(f"   Ensemble size: {config.ensemble_size}")
        print(f"   Quantiles: {config.num_quantiles}")
        print(f"   Action discretization: {config.action_discretization_bins} bins")
    
    def reset_storage(self):
        """Reset data storage - CONSISTENT WITH DISCRETE."""
        self.training_data = {
            # Core predictions
            "y_true": [],
            "y_pred": [],
            "states": [],
            "actions": [],
            
            # Uncertainty components
            "u_epistemic": [],
            "u_aleatoric": [],
            "u_total_method1_linear_addition": [],
            "u_total_method2_rss": [],
            "u_total_method3_dcor_entropy": [],
            "u_total_method4_nmi_entropy": [],
            "u_total_method5_upper_dcor": [],
            "u_total_method6_upper_nmi": [],
            
            # Raw outputs for post-processing
            "ensemble_predictions": [],
            "quantile_predictions": [],
            
            # Correlation metrics
            "correlation_pearson": [],
            "correlation_spearman": [],
            "correlation_distance": [],
            "correlation_nmi": [],
            
            # Metadata
            "episode_ids": [],
            "timesteps": [],
            "environment": "",
            "policy_type": "",
            "training_epochs": []
        }
    
    def load_dataset(self, env_name: str, policy_type: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """Load MuJoCo dataset - CONTINUOUS SPECIFIC LOADING."""
        print(f"Loading continuous dataset: {env_name}/{policy_type}")
        
        try:
            # Construct dataset path
            dataset_path = Path(self.config.dataset_path) / env_name / f"{policy_type}-v0" / "data" / "main_data.hdf5"
            
            if not dataset_path.exists():
                raise FileNotFoundError(f"Dataset not found: {dataset_path}")
            
            observations = []
            actions = []
            rewards = []
            episode_ids = []
            timesteps = []
            
            print(f"   Loading from: {dataset_path}")
            
            # Load HDF5 data
            with h5py.File(dataset_path, 'r') as f:
                # Get episode structure
                episode_keys = [key for key in f.keys() if key.startswith('episode_')]
                print(f"   Total episodes: {len(episode_keys)}")
                
                for ep_idx, ep_key in enumerate(episode_keys):
                    if len(observations) >= self.config.max_samples:
                        break
                    
                    episode = f[ep_key]
                    
                    # Extract episode data
                    ep_obs = episode['observations'][:]
                    ep_actions = episode['actions'][:]
                    ep_rewards = episode['rewards'][:] if 'rewards' in episode else np.zeros(len(ep_obs))
                    
                    observations.extend(ep_obs)
                    actions.extend(ep_actions)
                    rewards.extend(ep_rewards)
                    episode_ids.extend([ep_idx] * len(ep_obs))
                    timesteps.extend(list(range(len(ep_obs))))
            
            # Convert to numpy arrays and limit samples
            observations = np.array(observations[:self.config.max_samples])
            actions = np.array(actions[:self.config.max_samples])
            rewards = np.array(rewards[:self.config.max_samples])
            episode_ids = np.array(episode_ids[:self.config.max_samples])
            timesteps = np.array(timesteps[:self.config.max_samples])
            
            # Normalize observations - CONSISTENT WITH DISCRETE
            observations = (observations - observations.mean(axis=0)) / (observations.std(axis=0) + 1e-8)
            
            # Setup action discretizer
            action_dim = actions.shape[1]
            self.action_discretizer = ActionDiscretizer(
                action_dim=action_dim,
                bins=self.config.action_discretization_bins,
                action_bounds=(-1.0, 1.0)  # Assume normalized actions
            )
            
            # Discretize actions for QRDQN
            discrete_actions = self.action_discretizer.batch_discretize(actions)
            
            metadata = {
                'num_samples': len(observations),
                'obs_dim': observations.shape[1],
                'action_dim': action_dim,
                'num_discrete_actions': self.action_discretizer.total_discrete_actions,
                'env_name': env_name,
                'policy_type': policy_type,
                'episode_ids': episode_ids,
                'timesteps': timesteps,
                'discrete_actions': discrete_actions
            }
            
            print(f"   Loaded {len(observations)} samples")
            print(f"   Observation dim: {observations.shape[1]}")
            print(f"   Action dim: {action_dim}")
            print(f"   Discrete actions: {metadata['num_discrete_actions']}")
            
            return observations, actions, rewards, metadata
            
        except Exception as e:
            print(f" Error loading dataset {env_name}/{policy_type}: {e}")
            raise

class ConsistentContinuousTrainer:
    """Consistent continuous uncertainty trainer with loss functions."""
    
    def __init__(self, config: ConsistentContinuousConfig):
        self.config = config
        self.device = config.device
        
        # Initialize proper loss functions
        self.loss_functions = LossFunctions(gamma=0.99, device=self.device)
        self.target_manager = TargetNetworkManager(update_frequency=1000)
        
        # Create results directory
        Path(config.results_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize data storage - CONSISTENT WITH DISCRETE
        self.reset_storage()
        
        print(f" Consistent Continuous Uncertainty Framework initialized")
        print(f"   Device: {self.device}")
        print(f"   Results: {config.results_dir}")
        print(f"   Architecture: {config.hidden_dims}")
        print(f"   Ensemble size: {config.ensemble_size}")
        print(f"   Quantiles: {config.num_quantiles}")
        print(f"   Action discretization: {config.action_discretization_bins} bins")
    
    def reset_storage(self):
        """Reset data storage - CONSISTENT WITH DISCRETE."""
        self.training_data = {
            # Core predictions
            "y_true": [],
            "y_pred": [],
            "states": [],
            "actions": [],
            
            # Uncertainty components
            "u_epistemic": [],
            "u_aleatoric": [],
            "u_total_method1_linear_addition": [],
            "u_total_method2_rss": [],
            "u_total_method3_dcor_entropy": [],
            "u_total_method4_nmi_entropy": [],
            "u_total_method5_upper_dcor": [],
            "u_total_method6_upper_nmi": [],
            
            # Raw outputs for post-processing
            "ensemble_predictions": [],
            "quantile_predictions": [],
            
            # Correlation metrics
            "correlation_pearson": [],
            "correlation_spearman": [],
            "correlation_distance": [],
            "correlation_nmi": [],
            
            # Metadata
            "episode_ids": [],
            "timesteps": [],
            "environment": "",
            "policy_type": "",
            "training_epochs": []
        }
    
    def load_dataset(self, env_name: str, policy_type: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """Load MuJoCo dataset - CONTINUOUS SPECIFIC LOADING."""
        print(f"Loading continuous dataset: {env_name}/{policy_type}")
        
        try:
            # Construct dataset path
            dataset_path = Path(self.config.dataset_path) / env_name / f"{policy_type}-v0" / "data" / "main_data.hdf5"
            
            if not dataset_path.exists():
                raise FileNotFoundError(f"Dataset not found: {dataset_path}")
            
            observations = []
            actions = []
            rewards = []
            episode_ids = []
            timesteps = []
            
            print(f"   Loading from: {dataset_path}")
            
            # Load HDF5 data
            with h5py.File(dataset_path, 'r') as f:
                # Get episode structure
                episode_keys = [key for key in f.keys() if key.startswith('episode_')]
                print(f"   Total episodes: {len(episode_keys)}")
                
                for ep_idx, ep_key in enumerate(episode_keys):
                    if len(observations) >= self.config.max_samples:
                        break
                    
                    episode = f[ep_key]
                    
                    # Extract episode data
                    ep_obs = episode['observations'][:]
                    ep_actions = episode['actions'][:]
                    ep_rewards = episode['rewards'][:] if 'rewards' in episode else np.zeros(len(ep_obs))
                    
                    observations.extend(ep_obs)
                    actions.extend(ep_actions)
                    rewards.extend(ep_rewards)
                    episode_ids.extend([ep_idx] * len(ep_obs))
                    timesteps.extend(list(range(len(ep_obs))))
            
            # Convert to numpy arrays and limit samples
            observations = np.array(observations[:self.config.max_samples])
            actions = np.array(actions[:self.config.max_samples])
            rewards = np.array(rewards[:self.config.max_samples])
            episode_ids = np.array(episode_ids[:self.config.max_samples])
            timesteps = np.array(timesteps[:self.config.max_samples])
            
            # Normalize observations - CONSISTENT WITH DISCRETE
            observations = (observations - observations.mean(axis=0)) / (observations.std(axis=0) + 1e-8)
            
            # Setup action discretizer
            action_dim = actions.shape[1]
            self.action_discretizer = ActionDiscretizer(
                action_dim=action_dim,
                bins=self.config.action_discretization_bins,
                action_bounds=(-1.0, 1.0)  # Assume normalized actions
            )
            
            # Discretize actions for QRDQN
            discrete_actions = self.action_discretizer.batch_discretize(actions)
            
            metadata = {
                'num_samples': len(observations),
                'obs_dim': observations.shape[1],
                'action_dim': action_dim,
                'num_discrete_actions': self.action_discretizer.total_discrete_actions,
                'env_name': env_name,
                'policy_type': policy_type,
                'episode_ids': episode_ids,
                'timesteps': timesteps,
                'discrete_actions': discrete_actions
            }
            
            print(f"   Loaded {len(observations)} samples")
            print(f"   Observation dim: {observations.shape[1]}")
            print(f"   Action dim: {action_dim}")
            print(f"   Discrete actions: {metadata['num_discrete_actions']}")
            
            return observations, actions, rewards, metadata
            
        except Exception as e:
            print(f" Error loading dataset {env_name}/{policy_type}: {e}")
            raise

    def train_models_with_proper_losses(self, qrdqn: ContinuousQRDQN, ensemble: ContinuousEnsemble,
                                      states: np.ndarray, actions: np.ndarray, rewards: np.ndarray,
                                      next_states: np.ndarray, dones: np.ndarray) -> Dict[str, float]:
        """Train models using proper loss functions."""
        print(f" Training with proper loss functions")
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.FloatTensor(actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)
        dones_tensor = torch.FloatTensor(dones).to(self.device)
        
        # Discretize actions for QR-DQN (continuous-specific)
        action_discretizer = ActionDiscretizer(
            action_dim=actions.shape[1],
            num_bins=self.config.action_discretization_bins
        )
        discrete_actions = action_discretizer.discretize(actions_tensor)
        
        # Create target networks
        qrdqn_target = ContinuousQRDQN(
            input_size=states.shape[1],
            num_discrete_actions=self.config.action_discretization_bins ** actions.shape[1],
            num_quantiles=self.config.num_quantiles,
            hidden_dims=self.config.hidden_dims
        ).to(self.device)
        
        ensemble_targets = [
            ContinuousEnsembleMember(
                input_size=states.shape[1],
                output_size=actions.shape[1],
                hidden_dims=self.config.hidden_dims
            ).to(self.device) for _ in range(self.config.ensemble_size)
        ]
        
        # Initialize target networks
        self.target_manager.hard_update(qrdqn_target, qrdqn)
        for target_net, source_net in zip(ensemble_targets, ensemble.ensemble):
            self.target_manager.hard_update(target_net, source_net)
        
        # Set up optimizers
        qrdqn_optimizer = torch.optim.Adam(qrdqn.parameters(), lr=self.config.learning_rate)
        ensemble_optimizers = [
            torch.optim.Adam(member.parameters(), lr=self.config.learning_rate)
            for member in ensemble.ensemble
        ]
        
        # Create data loader
        dataset = TensorDataset(states_tensor, discrete_actions, rewards_tensor, 
                               next_states_tensor, dones_tensor)
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        
        training_metrics = {
            'qrdqn_losses': [],
            'ensemble_losses': [],
            'combined_losses': []
        }
        
        print(f"📊 Training for {self.config.num_epochs} epochs...")
        
        for epoch in range(self.config.num_epochs):
            epoch_qrdqn_losses = []
            epoch_ensemble_losses = []
            epoch_combined_losses = []
            
            for batch_idx, (batch_states, batch_discrete_actions, batch_rewards, 
                           batch_next_states, batch_dones) in enumerate(dataloader):
                
                # ===================
                # QR-DQN Training with Proper Loss
                # ===================
                qrdqn.train()
                qrdqn_target.eval()
                
                qrdqn_optimizer.zero_grad()
                
                # Use exact QR-DQN loss
                qrdqn_loss = self.loss_functions.qrdqn_loss_with_bellman(
                    qrdqn_network=qrdqn,
                    target_network=qrdqn_target,
                    states=batch_states,
                    actions=batch_discrete_actions,
                    rewards=batch_rewards,
                    next_states=batch_next_states,
                    dones=batch_dones,
                    quantile_fractions=qrdqn.quantile_fractions
                )
                
                qrdqn_loss.backward()
                torch.nn.utils.clip_grad_norm_(qrdqn.parameters(), 1.0)
                qrdqn_optimizer.step()
                
                epoch_qrdqn_losses.append(qrdqn_loss.item())
                
                # ===================
                # Ensemble Training with Proper Loss (Continuous-specific)
                # ===================
                # For continuous actions, we need to adapt the ensemble loss
                for member, target_member, optimizer in zip(
                    ensemble.ensemble, ensemble_targets, ensemble_optimizers
                ):
                    member.train()
                    target_member.eval()
                    
                    optimizer.zero_grad()
                    
                    # Continuous ensemble loss - predict action values
                    current_outputs = member(batch_states)  # [batch_size, action_dim]
                    
                    with torch.no_grad():
                        target_outputs = target_member(batch_next_states)  # [batch_size, action_dim]
                        # For continuous actions, use mean action value as target
                        targets = batch_rewards.unsqueeze(1) + self.loss_functions.gamma * (1 - batch_dones.unsqueeze(1)) * target_outputs.mean(dim=1, keepdim=True)
                    
                    # MSE loss between current outputs and targets
                    ensemble_loss = F.mse_loss(current_outputs.mean(dim=1, keepdim=True), targets)
                    
                    ensemble_loss.backward()
                    torch.nn.utils.clip_grad_norm_(member.parameters(), 1.0)
                    optimizer.step()
                    
                    epoch_ensemble_losses.append(ensemble_loss.item())
                
                # Combined loss for monitoring
                combined_loss = self.loss_functions.combined_training_objective(
                    qrdqn_loss=qrdqn_loss,
                    ensemble_loss=torch.tensor(np.mean(epoch_ensemble_losses[-self.config.ensemble_size:]))
                )
                epoch_combined_losses.append(combined_loss.item())
                
                # Update target networks periodically
                self.target_manager.update_if_needed(
                    target_networks=[qrdqn_target] + ensemble_targets,
                    source_networks=[qrdqn] + list(ensemble.ensemble)
                )
            
            # Log epoch metrics
            avg_qrdqn_loss = np.mean(epoch_qrdqn_losses)
            avg_ensemble_loss = np.mean(epoch_ensemble_losses)
            avg_combined_loss = np.mean(epoch_combined_losses)
            
            training_metrics['qrdqn_losses'].append(avg_qrdqn_loss)
            training_metrics['ensemble_losses'].append(avg_ensemble_loss)
            training_metrics['combined_losses'].append(avg_combined_loss)
            
            if (epoch + 1) % 3 == 0:
                print(f"   Epoch {epoch+1}/{self.config.num_epochs}: "
                      f"QR-DQN Loss: {avg_qrdqn_loss:.4f}, "
                      f"Ensemble Loss: {avg_ensemble_loss:.4f}, "
                      f"Combined: {avg_combined_loss:.4f}")
        
        print("Training completed!")
        return training_metrics
    
    def quantile_regression_loss(self, quantiles: torch.Tensor, targets: torch.Tensor,
                               quantile_fractions: torch.Tensor) -> torch.Tensor:
        """Use proper quantile regression loss implementation."""
        return self.loss_functions.quantile_regression_loss_exact(
            predicted_quantiles=quantiles,
            target_values=targets,
            quantile_fractions=quantile_fractions,
            use_huber=True,
            huber_delta=1.0
        )
    
    def compute_uncertainties(self, qrdqn: ContinuousQRDQN, ensemble: ContinuousEnsemble, 
                            observations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute uncertainties - CONSISTENT WITH DISCRETE."""
        print(f"🔬 Computing uncertainties...")
        
        obs_tensor = torch.FloatTensor(observations).to(self.device)
        
        # Compute uncertainties in batches
        batch_size = 512
        epistemic_uncertainties = []
        aleatoric_uncertainties = []
        
        for i in range(0, len(observations), batch_size):
            end_idx = min(i + batch_size, len(observations))
            batch_obs = obs_tensor[i:end_idx]
            
            # Epistemic uncertainty
            epistemic = ensemble.get_epistemic_uncertainty(batch_obs)
            epistemic_uncertainties.append(epistemic.cpu().numpy())
            
            # Aleatoric uncertainty
            aleatoric = qrdqn.get_aleatoric_uncertainty(batch_obs)
            aleatoric_uncertainties.append(aleatoric.cpu().numpy())
        
        epistemic_uncertainties = np.concatenate(epistemic_uncertainties)
        aleatoric_uncertainties = np.concatenate(aleatoric_uncertainties)
        
        print(f"   Epistemic uncertainty: {epistemic_uncertainties.mean():.4f} ± {epistemic_uncertainties.std():.4f}")
        print(f"   Aleatoric uncertainty: {aleatoric_uncertainties.mean():.4f} ± {aleatoric_uncertainties.std():.4f}")
        
        return epistemic_uncertainties, aleatoric_uncertainties
    
    def compute_uncertainty_combinations(self, epistemic: np.ndarray, aleatoric: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute the 6 uncertainty combination methods from the paper."""
        print(f"🔢 Computing 6 uncertainty combination methods...")
        
        methods = {}
        
        # (1) Independence-based Heuristics
        
        # Method 1: Linear Addition (Equation 294)
        methods['method1_linear_addition'] = epistemic + aleatoric
        
        # Method 2: Root-Sum-of-Squares (Equation 300)
        methods['method2_rss'] = np.sqrt(epistemic**2 + aleatoric**2)
        
        # (2) Correlation-Aware Combination
        
        # Compute correlation measures
        rho_dcor = self.distance_correlation(epistemic, aleatoric)
        rho_nmi = self.normalized_mutual_information(epistemic, aleatoric)
        
        # Compute joint entropy using KDE
        h_joint = self.compute_joint_entropy_kde(epistemic, aleatoric)
        
        # Method 3: Distance Correlation with Entropy Correction (Equation 310)
        combined_squared_dcor = epistemic**2 + aleatoric**2 - rho_dcor * h_joint
        methods['method3_dcor_entropy'] = np.sqrt(np.maximum(0, combined_squared_dcor))
        
        # Method 4: NMI with Entropy Correction (Equation 316)
        combined_squared_nmi = epistemic**2 + aleatoric**2 - rho_nmi * h_joint
        methods['method4_nmi_entropy'] = np.sqrt(np.maximum(0, combined_squared_nmi))
        
        # (3) Upper Bound Formulations with Theoretical Guarantees
        
        # Method 5: Upper Bound (dCor) (Equation 334)
        rss_term_dcor = (1 - rho_dcor) * np.sqrt(epistemic**2 + aleatoric**2)
        max_term_dcor = rho_dcor * np.maximum(epistemic, aleatoric)
        methods['method5_upper_dcor'] = rss_term_dcor + max_term_dcor
        
        # Method 6: Upper Bound (NMI) (Equation 338)
        rss_term_nmi = (1 - rho_nmi) * np.sqrt(epistemic**2 + aleatoric**2)
        max_term_nmi = rho_nmi * np.maximum(epistemic, aleatoric)
        methods['method6_upper_nmi'] = rss_term_nmi + max_term_nmi
        
        print(f"   Distance correlation (ρ_dCor): {rho_dcor:.4f}")
        print(f"   Normalized MI (ρ_NMI): {rho_nmi:.4f}")
        print(f"   Joint entropy (H_joint): {h_joint:.4f}")
        
        return methods
    
    def distance_correlation(self, x: np.ndarray, y: np.ndarray) -> float:
        """Distance correlation - CONSISTENT WITH DISCRETE."""
        if len(x) < 2:
            return 0.0
        
        n = len(x)
        
        # Compute pairwise distances
        a = np.abs(x[:, np.newaxis] - x[np.newaxis, :])
        b = np.abs(y[:, np.newaxis] - y[np.newaxis, :])
        
        # Center the distance matrices
        A = a - a.mean(axis=0)[np.newaxis, :] - a.mean(axis=1)[:, np.newaxis] + a.mean()
        B = b - b.mean(axis=0)[np.newaxis, :] - b.mean(axis=1)[:, np.newaxis] + b.mean()
        
        # Compute distance covariance and variances
        dcov_xy = np.sqrt(np.mean(A * B))
        dcov_xx = np.sqrt(np.mean(A * A))
        dcov_yy = np.sqrt(np.mean(B * B))
        
        if dcov_xx * dcov_yy == 0:
            return 0.0
        
        return dcov_xy / np.sqrt(dcov_xx * dcov_yy)
    
    def normalized_mutual_information(self, x: np.ndarray, y: np.ndarray) -> float:
        """Normalized mutual information - CONSISTENT WITH DISCRETE."""
        if len(x) < 2:
            return 0.0
        
        # Discretize continuous values
        bins = 10
        x_discrete = pd.cut(x, bins=bins, labels=False, duplicates='drop')
        y_discrete = pd.cut(y, bins=bins, labels=False, duplicates='drop')
        
        # Handle NaN values
        valid_mask = ~(pd.isna(x_discrete) | pd.isna(y_discrete))
        if valid_mask.sum() < 2:
            return 0.0
        
        from sklearn.metrics import mutual_info_score
        return mutual_info_score(x_discrete[valid_mask], y_discrete[valid_mask])
    
    def compute_mutual_information_kde(self, u_epi: np.ndarray, u_ale: np.ndarray) -> float:
        """Compute mutual information using KDE - CONSISTENT WITH DISCRETE."""
        if len(u_epi) < 10:
            return 0.0
        
        X = u_epi.reshape(-1, 1)
        y = u_ale
        
        try:
            mi = mutual_info_regression(X, y, random_state=42)[0]
            return max(0, mi)
        except:
            return 0.0
    
    def compute_joint_entropy_kde(self, x: np.ndarray, y: np.ndarray, bandwidth: float = None) -> float:
        """
        Compute joint entropy H(X,Y) using kernel density estimation.
        
        This implements the H_joint(s) term from equations 309-310 in the paper.
        """
        if len(x) < 2 or len(y) < 2:
            return 0.0
        
        try:
            from sklearn.neighbors import KernelDensity
            from sklearn.preprocessing import StandardScaler
            
            # Standardize the data
            xy_data = np.column_stack([x, y])
            scaler = StandardScaler()
            xy_scaled = scaler.fit_transform(xy_data)
            
            # Automatic bandwidth selection using Scott's rule
            if bandwidth is None:
                n, d = xy_scaled.shape
                bandwidth = n ** (-1. / (d + 4))
            
            # Fit KDE and estimate entropy
            kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
            kde.fit(xy_scaled)
            
            n_samples = min(1000, len(x))
            log_densities = kde.score_samples(xy_scaled[:n_samples])
            entropy = -np.mean(log_densities)
            
            return max(entropy, 0.0)
            
        except Exception as e:
            print(f"Warning: Joint entropy computation failed: {e}")
            return 0.0
    
    def analyze_correlations(self, epistemic: np.ndarray, aleatoric: np.ndarray) -> Dict[str, Any]:
        """Analyze correlations - CONSISTENT WITH DISCRETE."""
        print(f"📈 Analyzing correlations...")
        
        correlations = {}
        
        # Pearson correlation
        try:
            pearson_corr, pearson_p = pearsonr(epistemic, aleatoric)
            correlations['pearson'] = {'correlation': pearson_corr, 'p_value': pearson_p}
        except:
            correlations['pearson'] = {'correlation': 0.0, 'p_value': 1.0}
        
        # Spearman correlation
        try:
            spearman_corr, spearman_p = spearmanr(epistemic, aleatoric)
            correlations['spearman'] = {'correlation': spearman_corr, 'p_value': spearman_p}
        except:
            correlations['spearman'] = {'correlation': 0.0, 'p_value': 1.0}
        
        # Distance correlation
        correlations['distance'] = self.distance_correlation(epistemic, aleatoric)
        
        # Normalized mutual information
        correlations['nmi'] = self.normalized_mutual_information(epistemic, aleatoric)
        
        print(f"   Pearson: {correlations['pearson']['correlation']:.4f}")
        print(f"   Spearman: {correlations['spearman']['correlation']:.4f}")
        print(f"   Distance: {correlations['distance']:.4f}")
        print(f"   NMI: {correlations['nmi']:.4f}")
        
        return correlations
    
    def store_batch_results(self, states: np.ndarray, actions: np.ndarray, y_true: np.ndarray,
                          ensemble_preds: List[np.ndarray], quantile_preds: np.ndarray,
                          epistemic: np.ndarray, aleatoric: np.ndarray, methods: Dict[str, np.ndarray],
                          correlations: Dict[str, Any], episode_ids: np.ndarray, timesteps: np.ndarray,
                          environment: str, policy_type: str, epoch: int):
        """Store batch results - CONSISTENT WITH DISCRETE."""
        
        # Core predictions
        self.training_data["states"].extend(states)
        self.training_data["actions"].extend(actions)
        self.training_data["y_true"].extend(y_true)
        
        # Uncertainty components
        self.training_data["u_epistemic"].extend(epistemic)
        self.training_data["u_aleatoric"].extend(aleatoric)
        
        # All 6 uncertainty combination methods
        for method_name, values in methods.items():
            self.training_data[f"u_total_{method_name}"].extend(values)
        
        # Raw outputs
        self.training_data["ensemble_predictions"].extend(ensemble_preds)
        self.training_data["quantile_predictions"].extend(quantile_preds)
        
        # Correlation metrics
        self.training_data["correlation_pearson"].extend([correlations['pearson']['correlation']] * len(states))
        self.training_data["correlation_spearman"].extend([correlations['spearman']['correlation']] * len(states))
        self.training_data["correlation_distance"].extend([correlations['distance']] * len(states))
        self.training_data["correlation_nmi"].extend([correlations['nmi']] * len(states))
        
        # Metadata
        self.training_data["episode_ids"].extend(episode_ids)
        self.training_data["timesteps"].extend(timesteps)
        self.training_data["environment"] = environment
        self.training_data["policy_type"] = policy_type
        self.training_data["training_epochs"].extend([epoch] * len(states))
    
    def save_training_data(self, filename_prefix: str, format: str = "all"):
        """Save training data - CONSISTENT WITH DISCRETE."""
        print(f"💾 Saving training data...")
        
        results_dir = Path(self.config.results_dir)
        
        # Convert lists to numpy arrays
        data_dict = {}
        csv_compatible_data = {}
        
        for key, value in self.training_data.items():
            if isinstance(value, list) and len(value) > 0:
                data_dict[key] = np.array(value)
            else:
                data_dict[key] = value
        
        # Find the reference length (from epistemic uncertainty)
        reference_length = len(data_dict.get('u_epistemic', []))
        
        # Only include 1D arrays with matching length for CSV
        csv_keys = ['u_epistemic', 'u_aleatoric', 'u_total_method1_linear_addition', 'u_total_method2_rss', 
                   'u_total_method3_dcor_entropy', 'u_total_method4_nmi_entropy', 'u_total_method5_upper_dcor', 
                   'u_total_method6_upper_nmi', 'correlation_pearson', 'correlation_spearman', 
                   'correlation_distance', 'correlation_nmi', 'episode_ids', 'timesteps', 'training_epochs', 'y_true']
        
        for key in csv_keys:
            if key in data_dict and isinstance(data_dict[key], np.ndarray):
                if len(data_dict[key].shape) == 1 and len(data_dict[key]) == reference_length:
                    csv_compatible_data[key] = data_dict[key]
                elif len(data_dict[key].shape) == 1:
                    print(f"   ⚠️ Skipping {key} for CSV: length mismatch ({len(data_dict[key])} vs {reference_length})")
        
        # Add scalar metadata
        for key, value in data_dict.items():
            if isinstance(value, (str, int, float)):
                csv_compatible_data[f'meta_{key}'] = [value] * reference_length
        
        if format in ["all", "csv"]:
            # Save CSV with only compatible data
            try:
                if csv_compatible_data:
                    df = pd.DataFrame(csv_compatible_data)
                    csv_path = results_dir / f"{filename_prefix}.csv"
                    df.to_csv(csv_path, index=False)
                    print(f"   Saved CSV: {csv_path}")
                else:
                    print(f"   ⚠️ No CSV-compatible data found")
            except Exception as e:
                print(f"   ⚠️ CSV save failed: {e}")
        
        if format in ["all", "json"]:
            # Save metadata as JSON
            metadata = {
                'environment': data_dict.get('environment', ''),
                'policy_type': data_dict.get('policy_type', ''),
                'num_samples': len(data_dict.get('u_epistemic', [])),
                'ensemble_size': self.config.ensemble_size,
                'num_quantiles': self.config.num_quantiles,
                'hidden_dims': self.config.hidden_dims,
                'num_epochs': self.config.num_epochs,
                'batch_size': self.config.batch_size,
                'learning_rate': self.config.learning_rate,
                'action_discretization_bins': self.config.action_discretization_bins
            }
            
            json_path = results_dir / f"{filename_prefix}_metadata.json"
            with open(json_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"   Saved JSON metadata: {json_path}")
    
    def create_visualizations(self, epistemic: np.ndarray, aleatoric: np.ndarray, 
                            methods: Dict[str, np.ndarray], correlations: Dict[str, Any],
                            experiment_id: str) -> None:
        """Create visualizations - CONSISTENT WITH DISCRETE."""
        if not self.config.save_plots:
            return
        
        print(f"📊 Creating visualizations...")
        
        results_dir = Path(self.config.results_dir)
        

        plt.style.use('default')
        sns.set_palette("husl")
        
        #  Uncertainty scatter plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Epistemic vs Aleatoric
        axes[0, 0].scatter(epistemic, aleatoric, alpha=0.6, s=10)
        axes[0, 0].set_xlabel('Epistemic Uncertainty')
        axes[0, 0].set_ylabel('Aleatoric Uncertainty')
        axes[0, 0].set_title(f'Uncertainty Components\nPearson r={correlations["pearson"]["correlation"]:.3f}')
        
        # Method comparison
        method_names = ['method1_linear_addition', 'method2_rss', 'method5_upper_dcor', 'method6_upper_nmi']
        method_labels = ['Linear Addition', 'RSS', 'Upper Bound (dCor)', 'Upper Bound (NMI)']
        
        for i, (method, label) in enumerate(zip(method_names, method_labels)):
            if method in methods:
                row, col = divmod(i, 2)
                if row == 0 and col == 1:
                    axes[row, col].hist(methods[method], bins=30, alpha=0.7, label=label)
                    axes[row, col].set_xlabel('Total Uncertainty')
                    axes[row, col].set_ylabel('Frequency')
                    axes[row, col].set_title(f'{label} Method Distribution')
                elif row == 1:
                    axes[row, col].scatter(epistemic + aleatoric, methods[method], alpha=0.6, s=10)
                    axes[row, col].plot([0, max(epistemic + aleatoric)], [0, max(epistemic + aleatoric)], 'r--', alpha=0.5)
                    axes[row, col].set_xlabel('Sum Method')
                    axes[row, col].set_ylabel(f'{label} Method')
                    axes[row, col].set_title(f'Sum vs {label}')
        
        plt.tight_layout()
        plt.savefig(results_dir / f"{experiment_id}_uncertainty_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Correlation summary
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        correlation_values = [
            correlations['pearson']['correlation'],
            correlations['spearman']['correlation'],
            correlations['distance'],
            correlations['nmi']
        ]
        correlation_names = ['Pearson', 'Spearman', 'Distance', 'NMI']
        
        bars = ax.bar(correlation_names, correlation_values, alpha=0.7)
        ax.set_ylabel('Correlation Coefficient')
        ax.set_title(f'Uncertainty Correlation Analysis - {experiment_id}')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, correlation_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(results_dir / f"{experiment_id}_correlations.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   Saved visualizations to {results_dir}")
    
    def run_single_experiment(self, env_name: str, policy_type: str) -> Dict[str, Any]:
        """Run single experiment - CONSISTENT WITH DISCRETE."""
        print(f"\n🚀 Starting experiment: {env_name}/{policy_type}")
        
        try:
            # Reset storage for new experiment
            self.reset_storage()
            
            # Load dataset
            observations, actions, rewards, metadata = self.load_real_dataset(env_name, policy_type)
            
            # Train models
            qrdqn, ensemble = self.train_uncertainty_models(observations, actions, rewards, metadata)
            
            # Compute uncertainties
            epistemic, aleatoric = self.compute_uncertainties(qrdqn, ensemble, observations)
            
            # Compute uncertainty combinations
            methods = self.compute_uncertainty_combinations(epistemic, aleatoric)
            
            # Analyze correlations
            correlations = self.analyze_correlations(epistemic, aleatoric)
            
            # Create experiment ID
            experiment_id = f"{env_name}_{policy_type}"
            
            # Get ensemble predictions for storage
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(observations).to(self.device)
                ensemble_outputs = ensemble(obs_tensor)
                ensemble_preds = [output.cpu().numpy() for output in ensemble_outputs]
                
                quantile_outputs = qrdqn(obs_tensor)
                quantile_preds = quantile_outputs.cpu().numpy()
            
            # Store results
            self.store_batch_results(
                states=observations,
                actions=actions,
                y_true=rewards,
                ensemble_preds=ensemble_preds,
                quantile_preds=quantile_preds,
                epistemic=epistemic,
                aleatoric=aleatoric,
                methods=methods,
                correlations=correlations,
                episode_ids=metadata['episode_ids'],
                timesteps=metadata['timesteps'],
                environment=env_name,
                policy_type=policy_type,
                epoch=self.config.num_epochs
            )
            
            # Save data in essential formats only (avoid disk quota issues)
            self.save_training_data(f"continuous_{experiment_id}", format="csv")
            
            # Create visualizations
            self.create_visualizations(epistemic, aleatoric, methods, correlations, experiment_id)
            
            # Prepare results summary
            results = {
                'env_name': env_name,
                'policy_type': policy_type,
                'num_samples': len(observations),
                'correlations': correlations,
                'uncertainty_stats': {
                    'epistemic_mean': float(epistemic.mean()),
                    'epistemic_std': float(epistemic.std()),
                    'aleatoric_mean': float(aleatoric.mean()),
                    'aleatoric_std': float(aleatoric.std())
                },
                'method_stats': {
                    method: {
                        'mean': float(values.mean()),
                        'std': float(values.std())
                    } for method, values in methods.items()
                }
            }
            
            print(f" Experiment completed successfully: {env_name}/{policy_type}")
            return results
            
        except Exception as e:
            print(f" Experiment failed: {env_name}/{policy_type}")
            print(f"   Error: {str(e)}")
            traceback.print_exc()
            return {'env_name': env_name, 'policy_type': policy_type, 'error': str(e)}

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Consistent Continuous Uncertainty Correlation Study')
    parser.add_argument('--env_name', type=str, required=True,
                       help='Environment name (e.g., halfcheetah)')
    parser.add_argument('--policy_type', type=str, required=True,
                       help='Policy type (e.g., expert)')
    parser.add_argument('--results_dir', type=str, 
                       default='/var/scratch/zwa212/UQ_continuous_consistent_results',
                       help='Results directory')
    parser.add_argument('--epochs', type=int, default=15,
                       help='Number of training epochs')
    parser.add_argument('--ensemble_size', type=int, default=3,
                       help='Ensemble size')
    parser.add_argument('--quantiles', type=int, default=21,
                       help='Number of quantiles')
    parser.add_argument('--max_samples', type=int, default=10000,
                       help='Maximum samples to use')
    parser.add_argument('--action_bins', type=int, default=7,
                       help='Action discretization bins')
    
    return parser.parse_args()

def get_all_experiments():
    """Get all available continuous experiments."""
    environments = ["halfcheetah", "hopper", "walker2d"]
    policies = ["expert", "medium", "simple"]
    experiments = []
    for env in environments:
        for policy in policies:
            experiments.append((env, policy))
    return experiments

def main():
    """Main execution function."""
    print(" Consistent Continuous Uncertainty Correlation Study")
    print("=" * 60)
    
    # Parse arguments
    args = parse_arguments()
    
    # Create configuration
    config = ConsistentContinuousConfig(
        results_dir=args.results_dir,
        num_epochs=args.epochs,
        ensemble_size=args.ensemble_size,
        num_quantiles=args.quantiles,
        max_samples=args.max_samples,
        action_discretization_bins=args.action_bins
    )
    
    # Initialize framework
    framework = ConsistentContinuousUncertaintyFramework(config)
    
    # Run experiment
    results = framework.run_single_experiment(args.env_name, args.policy_type)
    
    # Save final results
    results_file = Path(config.results_dir) / "experiment_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n Results saved to: {results_file}")
    print(" Experiment completed!")

if __name__ == "__main__":
    main() 
 