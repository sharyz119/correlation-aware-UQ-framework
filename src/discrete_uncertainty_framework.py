#!/usr/bin/env python3
"""
Discrete Environment Uncertainty Correlation Study 
======================================================================

Complete implementation of uncertainty correlation analysis for discrete action spaces,
consistent with continuous implementation except for necessary environment differences.



Author: Zihan Wang
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

# Import Minari for real dataset loading
try:
    import minari
    MINARI_AVAILABLE = True
    print(" Minari imported successfully")
except ImportError:
    MINARI_AVAILABLE = False
    print(" Minari not available - install with: !pip install minari")

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Import proper loss functions
from loss_functions import LossFunctions, TargetNetworkManager

@dataclass
class ConsistentDiscreteConfig:
    """Consistent configuration for discrete uncertainty experiments."""
    # Training parameters - CONSISTENT WITH CONTINUOUS
    num_epochs: int = 15
    batch_size: int = 128
    learning_rate: float = 1e-4
    max_samples: int = 10000
    
    # Model parameters - CONSISTENT WITH CONTINUOUS
    ensemble_size: int = 3
    num_quantiles: int = 21
    hidden_dims: List[int] = None  # Will be set to [128, 128]
    
    # Analysis parameters - CONSISTENT WITH CONTINUOUS
    correlation_threshold: float = 0.3
    nmi_cap: float = 0.8
    
    # Environment parameters
    atari_image_size: int = 84
    
    # Output parameters
    results_dir: str = "/var/scratch/zwa212/UQ_discrete_consistent_results"
    save_plots: bool = True
    save_models: bool = False  # Consistent with continuous
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [128, 128]  # Consistent with continuous

class DiscreteQRDQN(nn.Module):
    """Quantile Regression DQN for discrete action spaces - CONSISTENT ARCHITECTURE."""
    
    def __init__(self, input_size: int, num_actions: int, num_quantiles: int = 21, hidden_dims: List[int] = None):
        super().__init__()
        self.input_size = input_size
        self.num_actions = num_actions
        self.num_quantiles = num_quantiles
        
        if hidden_dims is None:
            hidden_dims = [128, 128]
        
        # Network architecture - CONSISTENT WITH CONTINUOUS
        layers = []
        prev_dim = input_size
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)  # Consistent dropout rate
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_actions * num_quantiles))
        
        self.network = nn.Sequential(*layers)
        
        # Quantile fractions - CONSISTENT WITH CONTINUOUS
        self.register_buffer('quantile_fractions', 
                           torch.linspace(0.1, 0.9, num_quantiles))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning quantile values."""
        batch_size = x.size(0)
        output = self.network(x)
        return output.view(batch_size, self.num_actions, self.num_quantiles)
    
    def get_aleatoric_uncertainty(self, x: torch.Tensor) -> torch.Tensor:
        """Compute aleatoric uncertainty from quantile spread - CONSISTENT METHOD."""
        with torch.no_grad():
            quantiles = self.forward(x)
            # Use interquartile range as uncertainty measure - CONSISTENT
            q75 = torch.quantile(quantiles, 0.75, dim=-1)
            q25 = torch.quantile(quantiles, 0.25, dim=-1)
            uncertainty = q75 - q25
            return uncertainty.mean(dim=-1)  # Average over actions

class DiscreteEnsemble(nn.Module):
    """Deep ensemble for epistemic uncertainty - CONSISTENT ARCHITECTURE."""
    
    def __init__(self, input_size: int, num_actions: int, ensemble_size: int = 3, hidden_dims: List[int] = None):
        super().__init__()
        self.ensemble_size = ensemble_size
        self.num_actions = num_actions
        
        if hidden_dims is None:
            hidden_dims = [128, 128]
        
        # Create ensemble members - CONSISTENT WITH CONTINUOUS
        self.ensemble = nn.ModuleList()
        for i in range(ensemble_size):
            layers = []
            prev_dim = input_size
            
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
            
            # Output layer
            layers.append(nn.Linear(prev_dim, num_actions))
            
            member = nn.Sequential(*layers)
            self.ensemble.append(member)
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass through all ensemble members."""
        outputs = []
        for member in self.ensemble:
            outputs.append(member(x))
        return outputs
    
    def get_epistemic_uncertainty(self, x: torch.Tensor) -> torch.Tensor:
        """Compute epistemic uncertainty from ensemble disagreement - CONSISTENT METHOD."""
        with torch.no_grad():
            outputs = self.forward(x)
            stacked = torch.stack(outputs, dim=0)  # [ensemble_size, batch_size, num_actions]
            
            # Use variance across ensemble as epistemic uncertainty
            uncertainty = torch.var(stacked, dim=0)  # [batch_size, num_actions]
            return uncertainty.mean(dim=-1)  # Average over actions

class ConsistentDiscreteUncertaintyFramework:
    """Main framework for discrete uncertainty correlation analysis - CONSISTENT VERSION."""
    
    def __init__(self, config: ConsistentDiscreteConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Create results directory
        Path(config.results_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize data storage - CONSISTENT WITH CONTINUOUS
        self.reset_storage()
        
        print(f"🎮 Consistent Discrete Uncertainty Framework initialized")
        print(f"   Device: {self.device}")
        print(f"   Results: {config.results_dir}")
        print(f"   Architecture: {config.hidden_dims}")
        print(f"   Ensemble size: {config.ensemble_size}")
        print(f"   Quantiles: {config.num_quantiles}")
    
    def reset_storage(self):
        """Reset data storage - CONSISTENT WITH CONTINUOUS."""
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
    
    def load_dataset(self, dataset_id: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """Load real Minari dataset - CONSISTENT WITH CONTINUOUS LOADING PATTERN."""
        print(f" Loading discrete dataset: {dataset_id}")
        
        try:
            # Load dataset using Minari
            dataset = minari.load_dataset(dataset_id)
            
            observations = []
            actions = []
            rewards = []
            episode_ids = []
            timesteps = []
            
            print(f"   Total episodes: {len(dataset)}")
            
            # Process episodes
            for ep_id, episode in enumerate(dataset):
                if len(observations) >= self.config.max_samples:
                    break
                
                ep_obs = episode.observations
                ep_actions = episode.actions
                ep_rewards = episode.rewards
                
                # Flatten observations if needed (for Atari images)
                if len(ep_obs.shape) > 2:
                    ep_obs = ep_obs.reshape(ep_obs.shape[0], -1)
                
                observations.extend(ep_obs)
                actions.extend(ep_actions)
                rewards.extend(ep_rewards)
                episode_ids.extend([ep_id] * len(ep_obs))
                timesteps.extend(list(range(len(ep_obs))))
            
            # Convert to numpy arrays
            observations = np.array(observations[:self.config.max_samples])
            actions = np.array(actions[:self.config.max_samples])
            rewards = np.array(rewards[:self.config.max_samples])
            episode_ids = np.array(episode_ids[:self.config.max_samples])
            timesteps = np.array(timesteps[:self.config.max_samples])
            
            # Normalize observations
            observations = (observations - observations.mean(axis=0)) / (observations.std(axis=0) + 1e-8)
            
            metadata = {
                'num_samples': len(observations),
                'obs_dim': observations.shape[1],
                'num_actions': len(np.unique(actions)),
                'dataset_id': dataset_id,
                'episode_ids': episode_ids,
                'timesteps': timesteps
            }
            
            print(f"   Loaded {len(observations)} samples")
            print(f"   Observation dim: {observations.shape[1]}")
            print(f"   Action space: {metadata['num_actions']}")
            
            return observations, actions, rewards, metadata
            
        except Exception as e:
            print(f" Error loading dataset {dataset_id}: {e}")
            raise

    def train_uncertainty_models(self, observations: np.ndarray, actions: np.ndarray, rewards: np.ndarray, 
                               metadata: Dict) -> Tuple[DiscreteQRDQN, DiscreteEnsemble]:
        """Train uncertainty models - CONSISTENT TRAINING PROCEDURE."""
        print(f" Training uncertainty models...")
        
        # Create models
        obs_dim = observations.shape[1]
        num_actions = metadata['num_actions']
        
        qrdqn = DiscreteQRDQN(
            input_size=obs_dim,
            num_actions=num_actions,
            num_quantiles=self.config.num_quantiles,
            hidden_dims=self.config.hidden_dims
        ).to(self.device)
        
        ensemble = DiscreteEnsemble(
            input_size=obs_dim,
            num_actions=num_actions,
            ensemble_size=self.config.ensemble_size,
            hidden_dims=self.config.hidden_dims
        ).to(self.device)
        
        # Prepare data
        obs_tensor = torch.FloatTensor(observations).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        
        # Create dataset and dataloader
        dataset = TensorDataset(obs_tensor, actions_tensor, rewards_tensor)
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        
        # Optimizers - CONSISTENT WITH CONTINUOUS
        qrdqn_optimizer = optim.Adam(qrdqn.parameters(), lr=self.config.learning_rate)
        ensemble_optimizer = optim.Adam(ensemble.parameters(), lr=self.config.learning_rate)
        
        # Training loop - CONSISTENT WITH CONTINUOUS
        for epoch in range(self.config.num_epochs):
            qrdqn_losses = []
            ensemble_losses = []
            
            for batch_obs, batch_actions, batch_rewards in dataloader:
                # Train QRDQN
                qrdqn_optimizer.zero_grad()
                quantiles = qrdqn(batch_obs)
                
                # Quantile regression loss
                quantile_targets = batch_rewards.unsqueeze(1).unsqueeze(2).expand(-1, num_actions, self.config.num_quantiles)
                quantile_loss = self.quantile_regression_loss(quantiles, quantile_targets, qrdqn.quantile_fractions)
                quantile_loss.backward()
                qrdqn_optimizer.step()
                qrdqn_losses.append(quantile_loss.item())
                
                # Train Ensemble
                ensemble_optimizer.zero_grad()
                ensemble_outputs = ensemble(batch_obs)
                
                # MSE loss for each ensemble member
                ensemble_loss = 0
                for output in ensemble_outputs:
                    member_loss = F.mse_loss(output.gather(1, batch_actions.unsqueeze(1)).squeeze(1), batch_rewards)
                    ensemble_loss += member_loss
                ensemble_loss /= len(ensemble_outputs)
                
                ensemble_loss.backward()
                ensemble_optimizer.step()
                ensemble_losses.append(ensemble_loss.item())
            
            if (epoch + 1) % 5 == 0:
                print(f"   Epoch {epoch+1}/{self.config.num_epochs}: "
                      f"QRDQN Loss: {np.mean(qrdqn_losses):.4f}, "
                      f"Ensemble Loss: {np.mean(ensemble_losses):.4f}")
        
        return qrdqn, ensemble
    
    def quantile_regression_loss(self, quantiles: torch.Tensor, targets: torch.Tensor, 
                               quantile_fractions: torch.Tensor) -> torch.Tensor:
        """Quantile regression loss - CONSISTENT WITH CONTINUOUS."""
        diff = targets - quantiles
        loss = torch.mean(torch.max(quantile_fractions * diff, (quantile_fractions - 1) * diff))
        return loss
    
    def compute_uncertainties(self, qrdqn: DiscreteQRDQN, ensemble: DiscreteEnsemble, 
                            observations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute uncertainties - CONSISTENT WITH CONTINUOUS."""
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
        """Compute the 6 uncertainty combination methods."""
        print(f" Computing 6 uncertainty combination methods...")
        
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
        """Distance correlation - CONSISTENT WITH CONTINUOUS."""
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
        """Normalized mutual information - CONSISTENT WITH CONTINUOUS."""
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
        """Compute mutual information using KDE - CONSISTENT WITH CONTINUOUS."""
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
        """Analyze correlations - CONSISTENT WITH CONTINUOUS."""
        print(f" Analyzing correlations...")
        
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
        """Store batch results - CONSISTENT WITH CONTINUOUS."""
        
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
        """Save training data - CONSISTENT WITH CONTINUOUS."""
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
                    print(f"    Skipping {key} for CSV: length mismatch ({len(data_dict[key])} vs {reference_length})")
        
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
                    print(f"    No CSV-compatible data found")
            except Exception as e:
                print(f"    CSV save failed: {e}")
        
        if format in ["all", "json"]:
     
            metadata = {
                'environment': data_dict.get('environment', ''),
                'policy_type': data_dict.get('policy_type', ''),
                'num_samples': len(data_dict.get('u_epistemic', [])),
                'ensemble_size': self.config.ensemble_size,
                'num_quantiles': self.config.num_quantiles,
                'hidden_dims': self.config.hidden_dims,
                'num_epochs': self.config.num_epochs,
                'batch_size': self.config.batch_size,
                'learning_rate': self.config.learning_rate
            }
            
            json_path = results_dir / f"{filename_prefix}_metadata.json"
            with open(json_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"   Saved JSON metadata: {json_path}")
    
    def create_visualizations(self, epistemic: np.ndarray, aleatoric: np.ndarray, 
                            methods: Dict[str, np.ndarray], correlations: Dict[str, Any],
                            experiment_id: str) -> None:
        """Create visualizations - CONSISTENT WITH CONTINUOUS."""
        if not self.config.save_plots:
            return
        
        print(f" Creating visualizations...")
        
        results_dir = Path(self.config.results_dir)
        
        # Set up plotting style
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
        
        #  Correlation summary
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
    
    def run_single_experiment(self, dataset_id: str) -> Dict[str, Any]:
        """Run single experiment - CONSISTENT WITH CONTINUOUS."""
        print(f"\n🚀 Starting experiment: {dataset_id}")
        
        try:
            # Reset storage for new experiment
            self.reset_storage()
            
            # Load dataset
            observations, actions, rewards, metadata = self.load_real_dataset(dataset_id)
            
            # Train models
            qrdqn, ensemble = self.train_uncertainty_models(observations, actions, rewards, metadata)
            
            # Compute uncertainties
            epistemic, aleatoric = self.compute_uncertainties(qrdqn, ensemble, observations)
            
            # Compute uncertainty combinations
            methods = self.compute_uncertainty_combinations(epistemic, aleatoric)
            
            # Analyze correlations
            correlations = self.analyze_correlations(epistemic, aleatoric)
            
            # Create experiment ID
            experiment_id = dataset_id.replace('/', '_').replace('-', '_')
            
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
                environment=dataset_id.split('/')[1] if '/' in dataset_id else dataset_id,
                policy_type=dataset_id.split('/')[-1] if '/' in dataset_id else 'unknown',
                epoch=self.config.num_epochs
            )
            
            # Save data in essential formats only (avoid disk quota issues)
            self.save_training_data(f"discrete_{experiment_id}", format="csv")
            
            # Create visualizations
            self.create_visualizations(epistemic, aleatoric, methods, correlations, experiment_id)
            
            # Prepare results summary
            results = {
                'dataset_id': dataset_id,
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
            
            print(f" Experiment completed successfully: {dataset_id}")
            return results
            
        except Exception as e:
            print(f" Experiment failed: {dataset_id}")
            print(f"   Error: {str(e)}")
            traceback.print_exc()
            return {'dataset_id': dataset_id, 'error': str(e)}

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Consistent Discrete Uncertainty Correlation Study')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset ID (e.g., atari/breakout/expert-v0)')
    parser.add_argument('--results_dir', type=str, 
                       default='/var/scratch/zwa212/UQ_discrete_consistent_results',
                       help='Results directory')
    parser.add_argument('--epochs', type=int, default=15,
                       help='Number of training epochs')
    parser.add_argument('--ensemble_size', type=int, default=3,
                       help='Ensemble size')
    parser.add_argument('--quantiles', type=int, default=21,
                       help='Number of quantiles')
    parser.add_argument('--max_samples', type=int, default=10000,
                       help='Maximum samples to use')
    
    return parser.parse_args()

def get_all_datasets():
    """Get all available discrete datasets."""
    datasets = [
        "atari/breakout/expert-v0",
        "atari/pong/expert-v0", 
        "atari/qbert/expert-v0"
    ]
    return datasets

def main():
    """Main execution function."""
    print(" Consistent Discrete Uncertainty Correlation Study")
    print("=" * 60)
    
    # Parse arguments
    args = parse_arguments()
    
    # Create configuration
    config = ConsistentDiscreteConfig(
        results_dir=args.results_dir,
        num_epochs=args.epochs,
        ensemble_size=args.ensemble_size,
        num_quantiles=args.quantiles,
        max_samples=args.max_samples
    )
    
    # Initialize framework
    framework = ConsistentDiscreteUncertaintyFramework(config)
    
    # Run experiment
    results = framework.run_single_experiment(args.dataset)
    
    # Save final results
    results_file = Path(config.results_dir) / "experiment_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n Results saved to: {results_file}")
    print(" Experiment completed!")

if __name__ == "__main__":
    main() 
 