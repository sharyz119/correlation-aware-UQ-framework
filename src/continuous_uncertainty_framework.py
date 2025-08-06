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


warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

from core.model import create_continuous_uncertainty_models, ContinuousQRDQN, QRDQNEnsemble, ActionDiscretizer
from core.loss_functions import LossFunctions, TargetNetworkManager

@dataclass
class ConsistentContinuousConfig:
    """Consistent configuration for continuous uncertainty experiments."""
    # training parameters - CONSISTENT WITH DISCRETE
    num_epochs: int = 15
    batch_size: int = 128
    learning_rate: float = 1e-4
    max_samples: int = 10000
    
    # model parameters - CONSISTENT WITH DISCRETE
    ensemble_size: int = 3
    num_quantiles: int = 21
    hidden_dims: List[int] = None  # Will be set to [128, 128]
    
    # action discretization for QRDQN - CONTINUOUS SPECIFIC
    action_discretization_bins: int = 7
    use_action_discretization: bool = True
    
    # analysis parameters - CONSISTENT WITH DISCRETE
    correlation_threshold: float = 0.3
    nmi_cap: float = 0.8
    
    # dataset parameters
    dataset_path: str = "/var/scratch/zwa212/.minari/datasets/mujoco"
    environments: List[str] = None
    policies: List[str] = None
    
    # output parameters
    results_dir: str = "/var/scratch/zwa212/UQ_continuous_consistent_results"
    save_plots: bool = True
    save_models: bool = False  # Consistent with discrete
    
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [128, 128]  # Consistent with discrete
        if self.environments is None:
            self.environments = ["halfcheetah", "hopper", "walker2d"]
        if self.policies is None:
            self.policies = ["expert", "medium", "simple"]

class ConsistentContinuousUncertaintyFramework:
    """Main framework for continuous uncertainty correlation analysis - CONSISTENT VERSION."""
    
    def __init__(self, config: ConsistentContinuousConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # create results directory
        Path(config.results_dir).mkdir(parents=True, exist_ok=True)
        
        # initialize data storage - CONSISTENT WITH DISCRETE
        self.reset_storage()
        
        print(f"Consistent Continuous Uncertainty Framework initialized")
        print(f"   Device: {self.device}")
        print(f"   Results: {config.results_dir}")
        print(f"   Architecture: {config.hidden_dims}")
        print(f"   Ensemble size: {config.ensemble_size}")
        print(f"   Quantiles: {config.num_quantiles}")
        print(f"   Action discretization: {config.action_discretization_bins} bins")
    
    def reset_storage(self):
        """Reset data storage - CONSISTENT WITH DISCRETE."""
        self.training_data = {
            # core predictions
            "y_true": [],
            "y_pred": [],
            "states": [],
            "actions": [],
            
            # uncertainty components
            "u_epistemic": [],
            "u_aleatoric": [],
            "u_total_method1_linear_addition": [],
            "u_total_method2_rss": [],
            "u_total_method3_dcor_entropy": [],
            "u_total_method4_nmi_entropy": [],
            "u_total_method5_upper_dcor": [],
            "u_total_method6_upper_nmi": [],
            
            # raw outputs for post-processing
            "ensemble_predictions": [],
            "quantile_predictions": [],
            
            # correlation metrics
            "correlation_pearson": [],
            "correlation_spearman": [],
            "correlation_distance": [],
            "correlation_nmi": [],
            
            # metadata
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
            dataset_path = Path(self.config.dataset_path) / env_name / f"{policy_type}-v0" / "data" / "main_data.hdf5"
            
            if not dataset_path.exists():
                raise FileNotFoundError(f"Dataset not found: {dataset_path}")
            
            observations = []
            actions = []
            rewards = []
            episode_ids = []
            timesteps = []
            print(f"   Loading from: {dataset_path}")
            
            # load data
            with h5py.File(dataset_path, 'r') as f:
                # get episode structure
                episode_keys = [key for key in f.keys() if key.startswith('episode_')]
                print(f"   Total episodes: {len(episode_keys)}")
                
                for ep_idx, ep_key in enumerate(episode_keys):
                    if len(observations) >= self.config.max_samples:
                        break
                    
                    episode = f[ep_key]
                    
                    # extract episode data
                    ep_obs = episode['observations'][:]
                    ep_actions = episode['actions'][:]
                    ep_rewards = episode['rewards'][:] if 'rewards' in episode else np.zeros(len(ep_obs))
                    
                    observations.extend(ep_obs)
                    actions.extend(ep_actions)
                    rewards.extend(ep_rewards)
                    episode_ids.extend([ep_idx] * len(ep_obs))
                    timesteps.extend(list(range(len(ep_obs))))
            
            # convert to numpy arrays and limit samples
            observations = np.array(observations[:self.config.max_samples])
            actions = np.array(actions[:self.config.max_samples])
            rewards = np.array(rewards[:self.config.max_samples])
            episode_ids = np.array(episode_ids[:self.config.max_samples])
            timesteps = np.array(timesteps[:self.config.max_samples])
            
            # normalize observations - CONSISTENT WITH DISCRETE
            observations = (observations - observations.mean(axis=0)) / (observations.std(axis=0) + 1e-8)
            
            metadata = {
                'num_samples': len(observations),
                'obs_dim': observations.shape[1],
                'action_dim': actions.shape[1],
                'env_name': env_name,
                'policy_type': policy_type,
                'episode_ids': episode_ids,
                'timesteps': timesteps
            }
            
            print(f"   Loaded {len(observations)} samples")
            print(f"   Observation dim: {observations.shape[1]}")
            print(f"   Action dim: {actions.shape[1]}")
            
            return observations, actions, rewards, metadata
            
        except Exception as e:
            print(f"Error loading dataset {env_name}/{policy_type}: {e}")
            raise

    def train_uncertainty_models(self, observations: np.ndarray, actions: np.ndarray, rewards: np.ndarray, 
                               metadata: Dict) -> Tuple[ContinuousQRDQN, QRDQNEnsemble, ActionDiscretizer]:
        """Train uncertainty models using the new QR-DQN ensemble structure."""
        print(f"Training uncertainty models...")
        
        # create models using factory function
        obs_dim = observations.shape[1]
        action_dim = metadata['action_dim']
        
        single_qrdqn, qrdqn_ensemble, action_discretizer = create_continuous_uncertainty_models(
            state_dim=obs_dim,
            action_dim=action_dim,
            ensemble_size=self.config.ensemble_size,
            num_quantiles=self.config.num_quantiles,
            hidden_dims=self.config.hidden_dims,
            action_discretization_bins=self.config.action_discretization_bins
        )
        
        # move to device
        single_qrdqn = single_qrdqn.to(self.device)
        qrdqn_ensemble = qrdqn_ensemble.to(self.device)
        
        # prepare data
        obs_tensor = torch.FloatTensor(observations).to(self.device)
        
        # discretize actions for training
        discrete_actions = action_discretizer.batch_discretize(actions)
        discrete_actions_tensor = torch.LongTensor(discrete_actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        
        # create dataset and dataloader
        dataset = TensorDataset(obs_tensor, discrete_actions_tensor, rewards_tensor)
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        
        # optimizers - train both single QR-DQN and ensemble
        single_optimizer = optim.Adam(single_qrdqn.parameters(), lr=self.config.learning_rate)
        ensemble_optimizer = optim.Adam(qrdqn_ensemble.parameters(), lr=self.config.learning_rate)
        
        # training loop
        for epoch in range(self.config.num_epochs):
            single_losses = []
            ensemble_losses = []
            
            for batch_obs, batch_discrete_actions, batch_rewards in dataloader:
                # train single QR-DQN (for aleatoric uncertainty)
                single_optimizer.zero_grad()
                single_quantiles = single_qrdqn(batch_obs)
                
                # quantile regression loss for single QR-DQN
                single_loss = self.quantile_regression_loss(
                    single_quantiles, batch_rewards, single_qrdqn.quantile_fractions, batch_discrete_actions
                )
                single_loss.backward()
                single_optimizer.step()
                single_losses.append(single_loss.item())
                
                # train QR-DQN ensemble (for epistemic uncertainty)
                ensemble_optimizer.zero_grad()
                ensemble_quantile_outputs = qrdqn_ensemble(batch_obs)  # list of quantile outputs
                
                # compute ensemble loss (average over all ensemble members)
                ensemble_loss = 0
                for quantile_output in ensemble_quantile_outputs:
                    member_loss = self.quantile_regression_loss(
                        quantile_output, batch_rewards, qrdqn_ensemble.qrdqn_ensemble[0].quantile_fractions, batch_discrete_actions
                    )
                    ensemble_loss += member_loss
                ensemble_loss /= len(ensemble_quantile_outputs)
                
                ensemble_loss.backward()
                ensemble_optimizer.step()
                ensemble_losses.append(ensemble_loss.item())
            
            if (epoch + 1) % 5 == 0:
                print(f"   Epoch {epoch+1}/{self.config.num_epochs}: "
                      f"Single QR-DQN Loss: {np.mean(single_losses):.4f}, "
                      f"Ensemble Loss: {np.mean(ensemble_losses):.4f}")
        
        return single_qrdqn, qrdqn_ensemble, action_discretizer
    
    def quantile_regression_loss(self, quantiles: torch.Tensor, targets: torch.Tensor, 
                               quantile_fractions: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Quantile regression loss for continuous actions."""
        batch_size = targets.size(0)
        num_actions = quantiles.size(1)
        num_quantiles = quantiles.size(2)
        
        # expand targets for all actions and quantiles
        targets_expanded = targets.unsqueeze(1).unsqueeze(2).expand(batch_size, num_actions, num_quantiles)
        
        # compute quantile loss
        diff = targets_expanded - quantiles
        loss = torch.mean(torch.max(quantile_fractions * diff, (quantile_fractions - 1) * diff))
        
        return loss
    
    def compute_uncertainties(self, single_qrdqn: ContinuousQRDQN, qrdqn_ensemble: QRDQNEnsemble, 
                            observations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute uncertainties using the new QR-DQN ensemble structure."""
        print(f"Computing uncertainties...")
        
        obs_tensor = torch.FloatTensor(observations).to(self.device)
        
        # compute uncertainties in batches
        batch_size = 512
        epistemic_uncertainties = []
        aleatoric_uncertainties = []
        
        for i in range(0, len(observations), batch_size):
            end_idx = min(i + batch_size, len(observations))
            batch_obs = obs_tensor[i:end_idx]
            
            # epistemic uncertainty from QR-DQN ensemble disagreement
            epistemic = qrdqn_ensemble.get_epistemic_uncertainty(batch_obs)
            epistemic_uncertainties.append(epistemic.cpu().numpy())
            
            # aleatoric uncertainty from single QR-DQN quantile spread
            aleatoric = single_qrdqn.get_aleatoric_uncertainty(batch_obs)
            aleatoric_uncertainties.append(aleatoric.cpu().numpy())
        
        epistemic_uncertainties = np.concatenate(epistemic_uncertainties)
        aleatoric_uncertainties = np.concatenate(aleatoric_uncertainties)
        
        print(f"   Epistemic uncertainty: {epistemic_uncertainties.mean():.4f} ± {epistemic_uncertainties.std():.4f}")
        print(f"   Aleatoric uncertainty: {aleatoric_uncertainties.mean():.4f} ± {aleatoric_uncertainties.std():.4f}")
        
        return epistemic_uncertainties, aleatoric_uncertainties
    
    def compute_uncertainty_combinations(self, epistemic: np.ndarray, aleatoric: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute the 6 uncertainty combination methods from the paper."""
        print(f" Computing 6 uncertainty combination methods...")
        
        methods = {}
        
        # (1) Independence-based Heuristics
        
        # Method 1: Linear Addition 
        methods['method1_linear_addition'] = epistemic + aleatoric
        
        # Method 2: Root-Sum-of-Squares 
        methods['method2_rss'] = np.sqrt(epistemic**2 + aleatoric**2)
        
        # (2) Correlation-Aware Combination
        
        # Compute correlation measures
        rho_dcor = self.distance_correlation(epistemic, aleatoric)
        rho_nmi = self.normalized_mutual_information(epistemic, aleatoric)
        
        # Compute joint entropy using KDE
        h_joint = self.compute_joint_entropy_kde(epistemic, aleatoric)
        
        # Method 3: Distance Correlation with Entropy Correction 
        combined_squared_dcor = epistemic**2 + aleatoric**2 - rho_dcor * h_joint
        methods['method3_dcor_entropy'] = np.sqrt(np.maximum(0, combined_squared_dcor))
        
        # Method 4: NMI with Entropy Correction 
        combined_squared_nmi = epistemic**2 + aleatoric**2 - rho_nmi * h_joint
        methods['method4_nmi_entropy'] = np.sqrt(np.maximum(0, combined_squared_nmi))
        
        # (3) Upper Bound Formulations with Theoretical Guarantees
        
        # Method 5: Upper Bound (dCor) 
        rss_term_dcor = (1 - rho_dcor) * np.sqrt(epistemic**2 + aleatoric**2)
        max_term_dcor = rho_dcor * np.maximum(epistemic, aleatoric)
        methods['method5_upper_dcor'] = rss_term_dcor + max_term_dcor
        
        # Method 6: Upper Bound (NMI) 
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
        
        # compute pairwise distances
        a = np.abs(x[:, np.newaxis] - x[np.newaxis, :])
        b = np.abs(y[:, np.newaxis] - y[np.newaxis, :])
        
        # center the distance matrices
        A = a - a.mean(axis=0)[np.newaxis, :] - a.mean(axis=1)[:, np.newaxis] + a.mean()
        B = b - b.mean(axis=0)[np.newaxis, :] - b.mean(axis=1)[:, np.newaxis] + b.mean()
        
        # compute distance covariance and variances
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
        
        # discretize continuous values
        bins = 10
        x_discrete = pd.cut(x, bins=bins, labels=False, duplicates='drop')
        y_discrete = pd.cut(y, bins=bins, labels=False, duplicates='drop')
        
        # handle NaN values
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
            
            # standardize the data
            xy_data = np.column_stack([x, y])
            scaler = StandardScaler()
            xy_scaled = scaler.fit_transform(xy_data)
            
            # automatic bandwidth selection using Scott's rule
            if bandwidth is None:
                n, d = xy_scaled.shape
                bandwidth = n ** (-1. / (d + 4))
            
            # fit KDE and estimate entropy
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
        
        # NMI
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
        
        # core predictions
        self.training_data["states"].extend(states)
        self.training_data["actions"].extend(actions)
        self.training_data["y_true"].extend(y_true)
        
        # uncertainty components
        self.training_data["u_epistemic"].extend(epistemic)
        self.training_data["u_aleatoric"].extend(aleatoric)
        
        # all 6 uncertainty combination methods
        for method_name, values in methods.items():
            self.training_data[f"u_total_{method_name}"].extend(values)
        
        # raw outputs
        self.training_data["ensemble_predictions"].extend(ensemble_preds)
        self.training_data["quantile_predictions"].extend(quantile_preds)
        
        # correlation metrics
        self.training_data["correlation_pearson"].extend([correlations['pearson']['correlation']] * len(states))
        self.training_data["correlation_spearman"].extend([correlations['spearman']['correlation']] * len(states))
        self.training_data["correlation_distance"].extend([correlations['distance']] * len(states))
        self.training_data["correlation_nmi"].extend([correlations['nmi']] * len(states))
        
        # metadata
        self.training_data["episode_ids"].extend(episode_ids)
        self.training_data["timesteps"].extend(timesteps)
        self.training_data["environment"] = environment
        self.training_data["policy_type"] = policy_type
        self.training_data["training_epochs"].extend([epoch] * len(states))
    
    def save_training_data(self, filename_prefix: str, format: str = "all"):
        """Save training data - CONSISTENT WITH DISCRETE."""
        print(f" Saving training data...")
        
        results_dir = Path(self.config.results_dir)
        
        # convert lists to numpy arrays
        data_dict = {}
        csv_compatible_data = {}
        
        for key, value in self.training_data.items():
            if isinstance(value, list) and len(value) > 0:
                data_dict[key] = np.array(value)
            else:
                data_dict[key] = value
        
        # find the reference length (from epistemic uncertainty)
        reference_length = len(data_dict.get('u_epistemic', []))
        
        # only include 1D arrays with matching length for CSV
        csv_keys = ['u_epistemic', 'u_aleatoric', 'u_total_method1_linear_addition', 'u_total_method2_rss', 
                   'u_total_method3_dcor_entropy', 'u_total_method4_nmi_entropy', 'u_total_method5_upper_dcor', 
                   'u_total_method6_upper_nmi', 'correlation_pearson', 'correlation_spearman', 
                   'correlation_distance', 'correlation_nmi', 'episode_ids', 'timesteps', 'training_epochs', 'y_true']
        
        for key in csv_keys:
            if key in data_dict and isinstance(data_dict[key], np.ndarray):
                if len(data_dict[key].shape) == 1 and len(data_dict[key]) == reference_length:
                    csv_compatible_data[key] = data_dict[key]
                elif len(data_dict[key].shape) == 1:
                    print(f"  Skipping {key} for CSV: length mismatch ({len(data_dict[key])} vs {reference_length})")
        
        # add scalar metadata
        for key, value in data_dict.items():
            if isinstance(value, (str, int, float)):
                csv_compatible_data[f'meta_{key}'] = [value] * reference_length
        
        if format in ["all", "csv"]:
            # save CSV with only compatible data
            try:
                if csv_compatible_data:
                    df = pd.DataFrame(csv_compatible_data)
                    csv_path = results_dir / f"{filename_prefix}.csv"
                    df.to_csv(csv_path, index=False)
                    print(f"   Saved CSV: {csv_path}")
                else:
                    print(f"  No CSV-compatible data found")
            except Exception as e:
                print(f"  CSV save failed: {e}")
        
        if format in ["all", "json"]:
            # save metadata as JSON
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
        
        #  uncertainty scatter plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Epistemic vs Aleatoric
        axes[0, 0].scatter(epistemic, aleatoric, alpha=0.6, s=10)
        axes[0, 0].set_xlabel('Epistemic Uncertainty')
        axes[0, 0].set_ylabel('Aleatoric Uncertainty')
        axes[0, 0].set_title(f'Uncertainty Components\nPearson r={correlations["pearson"]["correlation"]:.3f}')
        
        # method comparison
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
        
        # correlation summary
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
        
        # add value labels on bars
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
        print(f"\nStarting experiment: {env_name}/{policy_type}")
        
        try:
            # reset storage for new experiment
            self.reset_storage()
            
            # load dataset
            observations, actions, rewards, metadata = self.load_dataset(env_name, policy_type)
            
            # train models
            single_qrdqn, qrdqn_ensemble, action_discretizer = self.train_uncertainty_models(observations, actions, rewards, metadata)
            
            # compute uncertainties
            epistemic, aleatoric = self.compute_uncertainties(single_qrdqn, qrdqn_ensemble, observations)
            
            # compute uncertainty combinations
            methods = self.compute_uncertainty_combinations(epistemic, aleatoric)
            
            # analyze correlations
            correlations = self.analyze_correlations(epistemic, aleatoric)
            
            # create experiment ID
            experiment_id = f"{env_name}_{policy_type}"
            
            # get ensemble predictions for storage
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(observations).to(self.device)
                ensemble_outputs = qrdqn_ensemble(obs_tensor)
                ensemble_preds = [output.cpu().numpy() for output in ensemble_outputs]
                
                quantile_outputs = single_qrdqn(obs_tensor)
                quantile_preds = quantile_outputs.cpu().numpy()
            
            # store results
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
            
            # save data in essential formats only (avoid disk quota issues)
            self.save_training_data(f"continuous_{experiment_id}", format="csv")
            
            # create visualizations
            self.create_visualizations(epistemic, aleatoric, methods, correlations, experiment_id)
            
            # prepare results summary
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
            
            print(f"Experiment completed successfully: {env_name}/{policy_type}")
            return results
            
        except Exception as e:
            print(f"Experiment failed: {env_name}/{policy_type}")
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
    print("Consistent Continuous Uncertainty Correlation Study")
    print("=" * 60)
    
    # parse arguments
    args = parse_arguments()
    
    # create configuration
    config = ConsistentContinuousConfig(
        results_dir=args.results_dir,
        num_epochs=args.epochs,
        ensemble_size=args.ensemble_size,
        num_quantiles=args.quantiles,
        max_samples=args.max_samples,
        action_discretization_bins=args.action_bins
    )
    

    framework = ConsistentContinuousUncertaintyFramework(config) 
    results = framework.run_single_experiment(args.env_name, args.policy_type)
    
    # save final results
    results_file = Path(config.results_dir) / "experiment_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {results_file}")
    print("Experiment completed!")

if __name__ == "__main__":
    main() 
 
