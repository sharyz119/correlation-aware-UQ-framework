"""
Comprehensive evaluation metrics for uncertainty quantification.

Implements evaluation methods for uncertainty quantification in reinforcement learning:
- Calibration Analysis (ECE, MCE, Brier Score, Reliability Diagrams)
- Conformal Prediction with adaptive sliding window
- Statistical Testing (binomial tests, Bonferroni correction, Cohen's h)
- Bootstrap confidence intervals
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Any, Optional
from scipy import stats
from scipy.stats import binom
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings('ignore')


class CalibrationAnalyzer:
    """
    Implements calibration analysis methods for uncertainty evaluation.
    
    Provides methods for computing Expected Calibration Error (ECE),
    Maximum Calibration Error (MCE), Brier Score, and reliability diagrams.
    """
    
    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
    
    def expected_calibration_error(self, y_true: np.ndarray, y_prob: np.ndarray, 
                                 n_bins: int = None) -> float:
        """
        Compute Expected Calibration Error (ECE).
        
        ECE = Σ(|B_b|/n) * |acc(B_b) - conf(B_b)|
        
        Args:
            y_true: True binary labels [0, 1]
            y_prob: Predicted probabilities [0, 1]
            n_bins: Number of bins for calibration
            
        Returns:
            Expected Calibration Error
        """
        if n_bins is None:
            n_bins = self.n_bins
            
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def maximum_calibration_error(self, y_true: np.ndarray, y_prob: np.ndarray,
                                n_bins: int = None) -> float:
        """
        Compute Maximum Calibration Error (MCE).
        
        MCE = max_b |acc(B_b) - conf(B_b)|
        """
        if n_bins is None:
            n_bins = self.n_bins
            
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        max_calibration_error = 0.0
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            
            if in_bin.sum() > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                calibration_error = np.abs(avg_confidence_in_bin - accuracy_in_bin)
                max_calibration_error = max(max_calibration_error, calibration_error)
        
        return max_calibration_error
    
    def brier_score(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """
        Compute Brier Score.
        
        BS = (1/n) * Σ(y_prob - y_true)²
        """
        return np.mean((y_prob - y_true) ** 2)
    
    def reliability_diagram_data(self, y_true: np.ndarray, y_prob: np.ndarray,
                               n_bins: int = None) -> Dict[str, np.ndarray]:
        """Generate data for reliability diagrams."""
        if n_bins is None:
            n_bins = self.n_bins
            
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        bin_centers = []
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            
            if in_bin.sum() > 0:
                bin_centers.append((bin_lower + bin_upper) / 2)
                bin_accuracies.append(y_true[in_bin].mean())
                bin_confidences.append(y_prob[in_bin].mean())
                bin_counts.append(in_bin.sum())
            else:
                bin_centers.append((bin_lower + bin_upper) / 2)
                bin_accuracies.append(0.0)
                bin_confidences.append(0.0)
                bin_counts.append(0)
        
        return {
            'bin_centers': np.array(bin_centers),
            'bin_accuracies': np.array(bin_accuracies),
            'bin_confidences': np.array(bin_confidences),
            'bin_counts': np.array(bin_counts)
        }
    
    def plot_reliability_diagram(self, y_true: np.ndarray, y_prob: np.ndarray,
                               title: str = "Reliability Diagram", 
                               save_path: str = None) -> plt.Figure:
        """Create reliability diagram plot."""
        data = self.reliability_diagram_data(y_true, y_prob)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        mask = data['bin_counts'] > 0
        ax.plot(data['bin_confidences'][mask], data['bin_accuracies'][mask], 
               'o-', label='Reliability', linewidth=2, markersize=6)
        
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', alpha=0.7)
        
        ax.hist(y_prob, bins=self.n_bins, density=True, alpha=0.3, 
               color='lightblue', label='Prediction Distribution')
        
        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def comprehensive_calibration_analysis(self, y_true: np.ndarray, 
                                         y_prob: np.ndarray) -> Dict[str, float]:
        """Perform comprehensive calibration analysis."""
        return {
            'ece': self.expected_calibration_error(y_true, y_prob),
            'mce': self.maximum_calibration_error(y_true, y_prob),
            'brier_score': self.brier_score(y_true, y_prob)
        }


class ConformalPredictor:
    """
    Implements conformal prediction with adaptive sliding window.
    
    Provides distribution-free uncertainty estimates with coverage guarantees
    as described in the paper.
    """
    
    def __init__(self, alpha: float = 0.1, window_size: int = 1000):
        self.alpha = alpha
        self.window_size = window_size
        self.conformity_scores = []
    
    def compute_conformity_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Compute conformity scores using absolute residuals."""
        return np.abs(y_true - y_pred)
    
    def calibrate(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Calibrate the conformal predictor using calibration data."""
        scores = self.compute_conformity_score(y_true, y_pred)
        self.conformity_scores = list(scores)
    
    def predict_interval(self, y_pred: np.ndarray, alpha: float = None) -> Tuple[np.ndarray, np.ndarray]:
        """Generate prediction intervals with conformal guarantee."""
        if alpha is None:
            alpha = self.alpha
            
        if len(self.conformity_scores) == 0:
            raise ValueError("Must calibrate predictor first")
        
        quantile_level = 1 - alpha
        q_hat = np.quantile(self.conformity_scores, quantile_level)
        
        lower_bounds = y_pred - q_hat
        upper_bounds = y_pred + q_hat
        
        return lower_bounds, upper_bounds
    
    def adaptive_predict_interval(self, y_pred: np.ndarray, 
                                recent_scores: List[float] = None,
                                alpha: float = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Adaptive conformal prediction with sliding window (Equation 362).
        
        q̂_α^(t) = Quantile_{1-α}({s_i : i ∈ W_t})
        """
        if alpha is None:
            alpha = self.alpha
            
        if recent_scores is not None:
            window_scores = recent_scores[-self.window_size:]
        else:
            window_scores = self.conformity_scores[-self.window_size:]
        
        if len(window_scores) == 0:
            raise ValueError("No conformity scores available")
        
        quantile_level = 1 - alpha
        q_hat_adaptive = np.quantile(window_scores, quantile_level)
        
        lower_bounds = y_pred - q_hat_adaptive
        upper_bounds = y_pred + q_hat_adaptive
        
        return lower_bounds, upper_bounds
    
    def update_scores(self, y_true: float, y_pred: float):
        """Update conformity scores with new observation."""
        score = self.compute_conformity_score(np.array([y_true]), np.array([y_pred]))[0]
        self.conformity_scores.append(score)
        
        if len(self.conformity_scores) > self.window_size:
            self.conformity_scores.pop(0)
    
    def evaluate_coverage(self, y_true: np.ndarray, lower_bounds: np.ndarray, 
                         upper_bounds: np.ndarray) -> Dict[str, float]:
        """Evaluate coverage properties of prediction intervals."""
        in_interval = (y_true >= lower_bounds) & (y_true <= upper_bounds)
        coverage_rate = np.mean(in_interval)
        
        interval_widths = upper_bounds - lower_bounds
        mean_width = np.mean(interval_widths)
        
        efficiency = coverage_rate / mean_width if mean_width > 0 else 0.0
        
        return {
            'coverage_rate': coverage_rate,
            'mean_width': mean_width,
            'efficiency': efficiency,
            'n_samples': len(y_true)
        }


class StatisticalTester:
    """
    Implements statistical testing methods for uncertainty evaluation.
    
    Includes binomial tests with Bonferroni correction, Cohen's h effect size
    estimation, and bootstrap confidence intervals.
    """
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
    
    def binomial_coverage_test(self, coverage_rate: float, target_coverage: float,
                             n_samples: int, alpha: float = None) -> Dict[str, Any]:
        """
        Binomial test for coverage rate.
        
        Tests H0: coverage_rate = target_coverage vs H1: coverage_rate ≠ target_coverage
        """
        if alpha is None:
            alpha = self.alpha
            
        n_successes = int(coverage_rate * n_samples)
        
        p_value = 2 * min(
            binom.cdf(n_successes, n_samples, target_coverage),
            1 - binom.cdf(n_successes - 1, n_samples, target_coverage)
        )
        
        ci_lower, ci_upper = self._wilson_confidence_interval(n_successes, n_samples, alpha)
        
        return {
            'coverage_rate': coverage_rate,
            'target_coverage': target_coverage,
            'n_samples': n_samples,
            'n_successes': n_successes,
            'p_value': p_value,
            'significant': p_value < alpha,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'alpha': alpha
        }
    
    def bonferroni_correction(self, p_values: List[float], alpha: float = None) -> Dict[str, Any]:
        """Apply Bonferroni correction for multiple comparisons."""
        if alpha is None:
            alpha = self.alpha
            
        n_tests = len(p_values)
        corrected_alpha = alpha / n_tests
        
        significant = [p < corrected_alpha for p in p_values]
        
        return {
            'original_alpha': alpha,
            'corrected_alpha': corrected_alpha,
            'n_tests': n_tests,
            'p_values': p_values,
            'significant': significant,
            'n_significant': sum(significant)
        }
    
    def cohens_h_effect_size(self, p1: float, p2: float) -> float:
        """
        Compute Cohen's h effect size for proportions.
        
        h = 2 * (arcsin(√p1) - arcsin(√p2))
        """
        p1 = np.clip(p1, 0.001, 0.999)
        p2 = np.clip(p2, 0.001, 0.999)
        
        h = 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))
        return h
    
    def bootstrap_confidence_interval(self, data: np.ndarray, statistic_func: callable,
                                    n_bootstrap: int = 1000, confidence_level: float = 0.95) -> Tuple[float, float]:
        """Bootstrap confidence interval for a statistic."""
        bootstrap_statistics = []
        n_samples = len(data)
        
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(data, size=n_samples, replace=True)
            stat = statistic_func(bootstrap_sample)
            bootstrap_statistics.append(stat)
        
        bootstrap_statistics = np.array(bootstrap_statistics)
        
        alpha = 1 - confidence_level
        lower_percentile = 100 * (alpha / 2)
        upper_percentile = 100 * (1 - alpha / 2)
        
        ci_lower = np.percentile(bootstrap_statistics, lower_percentile)
        ci_upper = np.percentile(bootstrap_statistics, upper_percentile)
        
        return ci_lower, ci_upper
    
    def _wilson_confidence_interval(self, n_successes: int, n_samples: int, 
                                  alpha: float) -> Tuple[float, float]:
        """Wilson confidence interval for binomial proportion."""
        z = stats.norm.ppf(1 - alpha / 2)
        p_hat = n_successes / n_samples
        
        denominator = 1 + (z**2) / n_samples
        center = p_hat + (z**2) / (2 * n_samples)
        margin = z * np.sqrt((p_hat * (1 - p_hat) + (z**2) / (4 * n_samples)) / n_samples)
        
        ci_lower = (center - margin) / denominator
        ci_upper = (center + margin) / denominator
        
        return max(0, ci_lower), min(1, ci_upper)


class ComprehensiveEvaluator:
    """
    Comprehensive evaluation framework combining all evaluation methods.
    
    Integrates calibration analysis, conformal prediction, and statistical testing
    for complete uncertainty quantification evaluation.
    """
    
    def __init__(self, n_bins: int = 10, alpha: float = 0.05, 
                 window_size: int = 1000, n_bootstrap: int = 1000):
        self.calibration_analyzer = CalibrationAnalyzer(n_bins=n_bins)
        self.conformal_predictor = ConformalPredictor(alpha=alpha, window_size=window_size)
        self.statistical_tester = StatisticalTester(alpha=alpha)
        self.n_bootstrap = n_bootstrap
    
    def evaluate_uncertainty_method(self, y_true: np.ndarray, y_pred: np.ndarray,
                                  uncertainty: np.ndarray, 
                                  method_name: str = "Unknown") -> Dict[str, Any]:
        """
        Comprehensive evaluation of an uncertainty quantification method.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            uncertainty: Uncertainty estimates
            method_name: Name of the method being evaluated
            
        Returns:
            Comprehensive evaluation results
        """
        results = {'method_name': method_name}
        
        # Convert uncertainty to probabilities for calibration analysis
        y_prob = 1 - stats.norm.cdf(np.abs(y_true - y_pred), scale=uncertainty)
        y_binary = (np.abs(y_true - y_pred) <= uncertainty).astype(int)
        
        # Calibration analysis
        results['calibration'] = self.calibration_analyzer.comprehensive_calibration_analysis(
            y_binary, y_prob
        )
        
        # Conformal prediction analysis
        n_cal = len(y_true) // 2
        
        self.conformal_predictor.calibrate(y_true[:n_cal], y_pred[:n_cal])
        lower_bounds, upper_bounds = self.conformal_predictor.predict_interval(y_pred[n_cal:])
        
        results['conformal'] = self.conformal_predictor.evaluate_coverage(
            y_true[n_cal:], lower_bounds, upper_bounds
        )
        
        # Statistical testing
        coverage_rate = results['conformal']['coverage_rate']
        target_coverage = 1 - self.conformal_predictor.alpha
        n_test_samples = len(y_true) - n_cal
        
        results['statistical_test'] = self.statistical_tester.binomial_coverage_test(
            coverage_rate, target_coverage, n_test_samples
        )
        
        # Bootstrap confidence intervals for key metrics
        results['bootstrap_ci'] = {
            'ece_ci': self.statistical_tester.bootstrap_confidence_interval(
                y_prob, lambda x: self.calibration_analyzer.expected_calibration_error(y_binary, x),
                n_bootstrap=self.n_bootstrap
            ),
            'coverage_ci': self.statistical_tester.bootstrap_confidence_interval(
                np.random.binomial(1, coverage_rate, n_test_samples),
                lambda x: np.mean(x),
                n_bootstrap=self.n_bootstrap
            )
        }
        
        return results
    
    def compare_methods(self, results_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compare multiple uncertainty quantification methods."""
        if len(results_list) < 2:
            raise ValueError("Need at least 2 methods to compare")
        
        comparison = {
            'n_methods': len(results_list),
            'method_names': [r['method_name'] for r in results_list],
            'pairwise_comparisons': {}
        }
        
        ece_values = [r['calibration']['ece'] for r in results_list]
        coverage_rates = [r['conformal']['coverage_rate'] for r in results_list]
        
        for i in range(len(results_list)):
            for j in range(i + 1, len(results_list)):
                method1 = results_list[i]['method_name']
                method2 = results_list[j]['method_name']
                
                h_coverage = self.statistical_tester.cohens_h_effect_size(
                    coverage_rates[i], coverage_rates[j]
                )
                
                comparison['pairwise_comparisons'][f'{method1}_vs_{method2}'] = {
                    'ece_difference': ece_values[i] - ece_values[j],
                    'coverage_difference': coverage_rates[i] - coverage_rates[j],
                    'cohens_h_coverage': h_coverage,
                    'effect_size_interpretation': self._interpret_cohens_h(h_coverage)
                }
        
        p_values = [r['statistical_test']['p_value'] for r in results_list]
        bonferroni_results = self.statistical_tester.bonferroni_correction(p_values)
        comparison['multiple_testing'] = bonferroni_results
        
        return comparison
    
    def _interpret_cohens_h(self, h: float) -> str:
        """Interpret Cohen's h effect size."""
        abs_h = abs(h)
        if abs_h < 0.2:
            return "negligible"
        elif abs_h < 0.5:
            return "small"
        elif abs_h < 0.8:
            return "medium"
        else:
            return "large"
    
   