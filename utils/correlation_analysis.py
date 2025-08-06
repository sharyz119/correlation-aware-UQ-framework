"""
Correlation analysis and uncertainty combination methods for reinforcement learning.

"""

import numpy as np
import torch
from typing import Dict, Any, Tuple
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import pdist, squareform
from sklearn.feature_selection import mutual_info_regression
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class CorrelationAnalyzer:
    """
    Analyzes correlations between epistemic and aleatoric uncertainties.
    
    Implements distance correlation, mutual information, and joint entropy calculations
    for comprehensive dependency analysis between uncertainty types.
    """
    
    def __init__(self, nmi_cap: float = 0.8, correlation_threshold: float = 0.3):
        self.nmi_cap = nmi_cap
        self.correlation_threshold = correlation_threshold
        self.scaler = StandardScaler()
    
    def distance_correlation(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute distance correlation between two variables.
        
        Distance correlation captures both linear and nonlinear dependencies
        and is zero if and only if the variables are independent.
        
        Args:
            x: First variable (epistemic uncertainty)
            y: Second variable (aleatoric uncertainty)
            
        Returns:
            Distance correlation coefficient in [0, 1]
        """
        def _distance_matrix(z):
            n = len(z)
            z = z.reshape(-1, 1) if z.ndim == 1 else z
            distances = squareform(pdist(z, metric='euclidean'))
            return distances
        
        def _doubly_centered_matrix(distances):
            n = distances.shape[0]
            row_means = distances.mean(axis=1)
            col_means = distances.mean(axis=0)
            grand_mean = distances.mean()
            
            centered = distances - row_means[:, np.newaxis] - col_means + grand_mean
            return centered
        
        try:
            x = np.asarray(x).flatten()
            y = np.asarray(y).flatten()
            
            if len(x) != len(y) or len(x) < 2:
                return 0.0
                
            # remove any infinite or NaN values
            mask = np.isfinite(x) & np.isfinite(y)
            x, y = x[mask], y[mask]
            
            if len(x) < 2:
                return 0.0
            
            # compute distance matrices and apply double centering
            A = _distance_matrix(x)
            B = _distance_matrix(y)
            A_centered = _doubly_centered_matrix(A)
            B_centered = _doubly_centered_matrix(B)
            
            # compute distance correlation
            n = len(x)
            dcov_xy = np.sqrt(np.sum(A_centered * B_centered) / (n * n))
            dcov_xx = np.sqrt(np.sum(A_centered * A_centered) / (n * n))
            dcov_yy = np.sqrt(np.sum(B_centered * B_centered) / (n * n))
            
            if dcov_xx > 0 and dcov_yy > 0:
                dcor = dcov_xy / np.sqrt(dcov_xx * dcov_yy)
                return min(max(dcor, 0.0), 1.0)
            else:
                return 0.0
                
        except Exception as e:
            print(f"Warning: Distance correlation computation failed: {e}")
            return 0.0

    def normalized_mutual_information(self, x: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
        """
        Compute normalized mutual information between two variables.
        
        Args:
            x: First variable
            y: Second variable
            bins: Number of bins for discretization
            
        Returns:
            Normalized mutual information in [0, 1]
        """
        try:
            x = np.asarray(x).flatten()
            y = np.asarray(y).flatten()
            
            if len(x) != len(y) or len(x) < 2:
                return 0.0
            
            # remove infinite and NaN values
            mask = np.isfinite(x) & np.isfinite(y)
            x, y = x[mask], y[mask]
            
            if len(x) < 2:
                return 0.0
            
            # use sklearn's mutual_info_regression for robust estimation
            mi = mutual_info_regression(x.reshape(-1, 1), y, discrete_features=False)[0]
            
            # compute marginal entropies for normalization
            def entropy(data, bins):
                hist, _ = np.histogram(data, bins=bins, density=True)
                hist = hist[hist > 0]
                return -np.sum(hist * np.log2(hist + 1e-10))
            
            h_x = entropy(x, bins)
            h_y = entropy(y, bins)
            
            # NMI
            if h_x > 0 and h_y > 0:
                nmi = mi / np.sqrt(h_x * h_y)
                return min(max(nmi, 0.0), 1.0)
            else:
                return 0.0
                
        except Exception as e:
            print(f"Warning: NMI computation failed: {e}")
            return 0.0

    def compute_joint_entropy_kde(self, x: np.ndarray, y: np.ndarray, bandwidth: float = None) -> float:
        """
        Compute joint entropy H(X,Y) using kernel density estimation.
        
        This implements the H_joint(s) term from equations 309-310 in the paper.
        
        Args:
            x: First variable (epistemic uncertainty)
            y: Second variable (aleatoric uncertainty)
            bandwidth: KDE bandwidth (auto-selected if None)
            
        Returns:
            Joint entropy estimate
        """
        try:
            x = np.asarray(x).flatten()
            y = np.asarray(y).flatten()
            
            if len(x) != len(y) or len(x) < 2:
                return 0.0
            
            # remove infinite and NaN values
            mask = np.isfinite(x) & np.isfinite(y)
            x, y = x[mask], y[mask]
            
            if len(x) < 2:
                return 0.0
            
            # standardize the data
            xy_data = np.column_stack([x, y])
            xy_scaled = self.scaler.fit_transform(xy_data)
            
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

    def compute_mutual_information_kde(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute mutual information I(X,Y) using KDE.
        
        Uses the relationship: I(X,Y) = H(X) + H(Y) - H(X,Y)
        
        Args:
            x: First variable
            y: Second variable
            
        Returns:
            Mutual information estimate
        """
        try:
            h_x = self._compute_marginal_entropy_kde(x)
            h_y = self._compute_marginal_entropy_kde(y)
            h_xy = self.compute_joint_entropy_kde(x, y)
            
            mi = h_x + h_y - h_xy
            return max(mi, 0.0)
            
        except Exception as e:
            print(f"Warning: Mutual information computation failed: {e}")
            return 0.0
    
    def _compute_marginal_entropy_kde(self, x: np.ndarray, bandwidth: float = None) -> float:
        """Compute marginal entropy H(X) using KDE."""
        try:
            x = np.asarray(x).flatten()
            mask = np.isfinite(x)
            x = x[mask]
            
            if len(x) < 2:
                return 0.0
            
            x_scaled = self.scaler.fit_transform(x.reshape(-1, 1)).flatten()
            
            if bandwidth is None:
                bandwidth = len(x) ** (-1. / 5.)  # Scott's rule for 1D
            
            kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
            kde.fit(x_scaled.reshape(-1, 1))
            
            n_samples = min(500, len(x))
            log_densities = kde.score_samples(x_scaled[:n_samples].reshape(-1, 1))
            entropy = -np.mean(log_densities)
            
            return max(entropy, 0.0)
            
        except Exception:
            return 0.0

    def analyze_all_correlations(self, epistemic: np.ndarray, aleatoric: np.ndarray) -> Dict[str, Any]:
        """
        Perform comprehensive correlation analysis between epistemic and aleatoric uncertainties.
        
        Args:
            epistemic: Epistemic uncertainty values
            aleatoric: Aleatoric uncertainty values
            
        Returns:
            Dictionary containing all correlation metrics and joint entropy
        """
        try:
            pearson_r, pearson_p = pearsonr(epistemic, aleatoric)
            spearman_r, spearman_p = spearmanr(epistemic, aleatoric)
            
            dcor = self.distance_correlation(epistemic, aleatoric)
            nmi = self.normalized_mutual_information(epistemic, aleatoric)
            
            # joint entropy and mutual information
            joint_entropy = self.compute_joint_entropy_kde(epistemic, aleatoric)
            mutual_info = self.compute_mutual_information_kde(epistemic, aleatoric)
            
            return {
                'pearson_correlation': pearson_r,
                'pearson_p_value': pearson_p,
                'spearman_correlation': spearman_r,
                'spearman_p_value': spearman_p,
                'distance_correlation': dcor,
                'normalized_mutual_information': nmi,
                'joint_entropy': joint_entropy,
                'mutual_information': mutual_info,
                'sample_size': len(epistemic),
                'epistemic_stats': {
                    'mean': np.mean(epistemic),
                    'std': np.std(epistemic),
                    'min': np.min(epistemic),
                    'max': np.max(epistemic)
                },
                'aleatoric_stats': {
                    'mean': np.mean(aleatoric),
                    'std': np.std(aleatoric),
                    'min': np.min(aleatoric),
                    'max': np.max(aleatoric)
                }
            }
            
        except Exception as e:
            print(f"Warning: Correlation analysis failed: {e}")
            return self._get_default_correlation_results(epistemic, aleatoric)
    
    def _get_default_correlation_results(self, epistemic: np.ndarray, aleatoric: np.ndarray) -> Dict[str, Any]:
        """Return default correlation results when analysis fails."""
        return {
            'pearson_correlation': 0.0,
            'pearson_p_value': 1.0,
            'spearman_correlation': 0.0,
            'spearman_p_value': 1.0,
            'distance_correlation': 0.0,
            'normalized_mutual_information': 0.0,
            'joint_entropy': 0.0,
            'mutual_information': 0.0,
            'sample_size': len(epistemic),
            'epistemic_stats': {'mean': 0, 'std': 0, 'min': 0, 'max': 0},
            'aleatoric_stats': {'mean': 0, 'std': 0, 'min': 0, 'max': 0}
        }


class UncertaintyCombiner:
    """
    Implements the 6 uncertainty combination methods.
    
    1. Linear Addition
    2. Root-Sum-of-Squares (RSS)  
    3. Distance Correlation with Entropy Correction
    4. NMI with Entropy Correction
    5. Upper Bound (dCor)
    6. Upper Bound (NMI)
    """
    
    def __init__(self, nmi_cap: float = 0.8):
        self.nmi_cap = nmi_cap
        self.correlation_analyzer = CorrelationAnalyzer(nmi_cap=nmi_cap)

    def method1_linear_addition(self, epistemic: np.ndarray, aleatoric: np.ndarray) -> np.ndarray:
        """Linear Addition (Equation 294): u_add(s) = u_e(s) + u_a(s)"""
        return epistemic + aleatoric
    
    def method2_root_sum_squares(self, epistemic: np.ndarray, aleatoric: np.ndarray) -> np.ndarray:
        """Root-Sum-of-Squares (Equation 300): u_RSS(s) = √(u_e²(s) + u_a²(s))"""
        return np.sqrt(epistemic**2 + aleatoric**2)
    
    def method3_dcor_entropy_correction(self, epistemic: np.ndarray, aleatoric: np.ndarray) -> np.ndarray:
        """
        Distance Correlation with Entropy Correction:
        u_dCor(s) = √(max(0, u_e²(s) + u_a²(s) - ρ_dCor · H_joint(s)))
        """
        rho_dcor = self.correlation_analyzer.distance_correlation(epistemic, aleatoric)
        h_joint = self.correlation_analyzer.compute_joint_entropy_kde(epistemic, aleatoric)
        
        combined_squared = epistemic**2 + aleatoric**2 - rho_dcor * h_joint
        combined = np.sqrt(np.maximum(0, combined_squared))
        
        return combined
    
    def method4_nmi_entropy_correction(self, epistemic: np.ndarray, aleatoric: np.ndarray) -> np.ndarray:
        """
        NMI with Entropy Correction:
        u_NMI(s) = √(max(0, u_e²(s) + u_a²(s) - ρ_NMI · H_joint(s)))
        """
        rho_nmi = self.correlation_analyzer.normalized_mutual_information(epistemic, aleatoric)
        h_joint = self.correlation_analyzer.compute_joint_entropy_kde(epistemic, aleatoric)
        
        combined_squared = epistemic**2 + aleatoric**2 - rho_nmi * h_joint
        combined = np.sqrt(np.maximum(0, combined_squared))
        
        return combined
    
    def method5_upper_bound_dcor(self, epistemic: np.ndarray, aleatoric: np.ndarray) -> np.ndarray:
        """
        Upper Bound (dCor):
        U_upper_dCor = (1-ρ_dCor)√(U_e² + U_a²) + ρ_dCor·max(U_e, U_a)
        """
        rho_dcor = self.correlation_analyzer.distance_correlation(epistemic, aleatoric)
        
        rss_term = (1 - rho_dcor) * np.sqrt(epistemic**2 + aleatoric**2)
        max_term = rho_dcor * np.maximum(epistemic, aleatoric)
        
        return rss_term + max_term
    
    def method6_upper_bound_nmi(self, epistemic: np.ndarray, aleatoric: np.ndarray) -> np.ndarray:
        """
        Upper Bound (NMI):
        U_upper_NMI = (1-ρ_NMI)√(U_e² + U_a²) + ρ_NMI·max(U_e, U_a)
        """
        rho_nmi = self.correlation_analyzer.normalized_mutual_information(epistemic, aleatoric)
        
        rss_term = (1 - rho_nmi) * np.sqrt(epistemic**2 + aleatoric**2)
        max_term = rho_nmi * np.maximum(epistemic, aleatoric)
        
        return rss_term + max_term
    
    def compute_all_combinations(self, epistemic: np.ndarray, aleatoric: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute all 6 uncertainty combination methods.
        
        Args:
            epistemic: Epistemic uncertainty values
            aleatoric: Aleatoric uncertainty values
            
        Returns:
            Dictionary with all 6 combination methods
        """
        try:
            # convert to numpy arrays and ensure finite values
            epistemic = np.asarray(epistemic).flatten()
            aleatoric = np.asarray(aleatoric).flatten()
            
            # remove infinite and NaN values
            mask = np.isfinite(epistemic) & np.isfinite(aleatoric)
            epistemic = epistemic[mask]
            aleatoric = aleatoric[mask]
            
            if len(epistemic) == 0:
                return {f'method{i+1}': np.array([0.0]) for i in range(6)}
            
            # compute all methods
            methods = {
                'method1_linear_addition': self.method1_linear_addition(epistemic, aleatoric),
                'method2_rss': self.method2_root_sum_squares(epistemic, aleatoric),
                'method3_dcor_entropy': self.method3_dcor_entropy_correction(epistemic, aleatoric),
                'method4_nmi_entropy': self.method4_nmi_entropy_correction(epistemic, aleatoric),
                'method5_upper_dcor': self.method5_upper_bound_dcor(epistemic, aleatoric),
                'method6_upper_nmi': self.method6_upper_bound_nmi(epistemic, aleatoric)
            }
            
            return methods
            
        except Exception as e:
            print(f"Warning: Uncertainty combination failed: {e}")
            return {f'method{i+1}': np.zeros_like(epistemic) for i in range(6)}


class ComprehensiveUncertaintyAnalysis:
    """
    
    Combines correlation analysis and uncertainty combination with method
    recommendation based on the dependency structure between uncertainty types.
    """
    
    def __init__(self, nmi_cap: float = 0.8, correlation_threshold: float = 0.3):
        self.nmi_cap = nmi_cap
        self.correlation_threshold = correlation_threshold
        self.correlation_analyzer = CorrelationAnalyzer(nmi_cap=nmi_cap, correlation_threshold=correlation_threshold)
        self.uncertainty_combiner = UncertaintyCombiner(nmi_cap=nmi_cap)

    def analyze_uncertainty_correlation(self, epistemic: np.ndarray, aleatoric: np.ndarray) -> Dict[str, Any]:
        """
        Perform complete uncertainty correlation analysis.
        
        Args:
            epistemic: Epistemic uncertainty values
            aleatoric: Aleatoric uncertainty values
            
        Returns:
            Comprehensive analysis results including correlations, combinations, and recommendations
        """
        # basic correlation analysis
        correlation_results = self.correlation_analyzer.analyze_all_correlations(epistemic, aleatoric)
        # uncertainty combination methods
        combination_results = self.uncertainty_combiner.compute_all_combinations(epistemic, aleatoric)
        # method recommendation
        recommendation = self._recommend_method(correlation_results)
        # compile comprehensive results
        results = {
            'correlations': correlation_results,
            'combinations': combination_results,
            'recommendation': recommendation,
            'analysis_summary': {
                'strong_correlation': correlation_results['distance_correlation'] > self.correlation_threshold,
                'high_mutual_information': correlation_results['normalized_mutual_information'] > 0.2,
                'joint_entropy': correlation_results['joint_entropy'],
                'sample_size': correlation_results['sample_size']
            }
        }
        
        return results

    