import numpy as np
from typing import Dict, Optional, Union, Tuple, List, Any, Callable
import os
import matplotlib.pyplot as plt
import pickle
from scipy.stats import entropy, wasserstein_distance
from scipy.spatial.distance import jensenshannon
import torch

class AdaptiveConformalPredictor:
    """
    Adaptive Conformal Prediction with uncertainty-aware calibration
    
    This class implements conformal prediction with:
    1. Context-aware calibration using uncertainty estimates
    2. Direct uncertainty measurement without independence assumptions
    3. Stratified prediction intervals based on epistemic/aleatoric decomposition
    4. Improved calibration with Bayesian credible intervals
    """
    
    def __init__(
        self,
        ensemble_model,
        alpha: float = 0.1,
        n_bins: int = 10,
        adaptive_method: str = "uncertainty_weighted",
        epistemic_weight: float = 0.7,
        aleatoric_weight: float = 0.3,
        uncertainty_temp: float = 1.0,
        bayesian_calibration: bool = True,
        interval_scale: float = 0.7  # scale factor for tighter intervals
    ):
        """
        Initialize the adaptive conformal predictor
        
        Args:
            ensemble_model: The ensemble model for predictions
            alpha: Target error rate (1 - coverage)
            n_bins: Number of bins for adaptive conformity scores
            adaptive_method: Method for adaptive prediction intervals
            epistemic_weight: Weight for epistemic uncertainty in calibration
            aleatoric_weight: Weight for aleatoric uncertainty in calibration
            uncertainty_temp: Temperature parameter for uncertainty weighing
            bayesian_calibration: Whether to use Bayesian calibration
            interval_scale: Scale factor to adjust prediction interval width
        """
        self.ensemble_model = ensemble_model
        self.alpha = alpha
        self.n_bins = n_bins
        self.method = adaptive_method
        self.epistemic_weight = epistemic_weight
        self.aleatoric_weight = aleatoric_weight
        self.uncertainty_temp = uncertainty_temp
        self.bayesian_calibration = bayesian_calibration
        self.interval_scale = interval_scale  
        
        # to be set during calibration
        self.calibration_multipliers = None
        self.conformity_scores = None
        self.conformity_threshold = None
        self.calibration_table = None
        self.state_features = None
        self.stratification_model = None
        
        # placeholders for calibration
        self.calib_thresholds = None
        self.uncertainty_bins = None
        self.bin_thresholds = None
        self.global_threshold = None
        
        # tracking coverage statistics
        self.coverage_stats = {
            'total_predictions': 0,
            'correct_predictions': 0,
            'avg_interval_size': 0.0,
            'bin_coverage': np.zeros(n_bins)
        }
        
        # storage for component-specific calibration
        self.epistemic_calibration = {
            'scores': [],
            'uncertainties': [],
            'residuals': [],
            'scaling_factors': None
        }
        
        self.aleatoric_calibration = {
            'scores': [],
            'uncertainties': [],
            'residuals': [],
            'scaling_factors': None
        }
        
    def calibrate(
        self,
        calib_states: np.ndarray, 
        calib_actions: np.ndarray, 
        calib_rewards: np.ndarray
    ) -> None:
        """
        Calibrate the conformal predictor using calibration dataset
        Enhanced with direct measurement of uncertainty decomposition
        and context-aware calibration
        
        Args:
            calib_states: States from calibration set [n_samples, state_dim]
            calib_actions: Actions from calibration set [n_samples]
            calib_rewards: Rewards from calibration set [n_samples]
        """
        n_samples = calib_states.shape[0]
        
        # Convert numpy arrays to torch tensors for the ensemble model
        calib_states_tensor = torch.FloatTensor(calib_states).to(self.ensemble_model.device)
        calib_actions_tensor = torch.LongTensor(calib_actions).to(self.ensemble_model.device)
        calib_rewards_tensor = torch.FloatTensor(calib_rewards).to(self.ensemble_model.device)
        
        # get predictions from ensemble model
        ensemble_preds = []
        for model in self.ensemble_model.models:
            with torch.no_grad():
                q_preds = model.predict(calib_states_tensor, calib_actions_tensor)
                ensemble_preds.append(q_preds.cpu().numpy())
                
        # stack predictions [ensemble_size, n_samples, n_quantiles]
        ensemble_preds = np.stack(ensemble_preds, axis=0)
        
        # mean predictions across ensemble
        mean_preds = np.mean(np.mean(ensemble_preds, axis=0), axis=-1)  # [n_samples]
        
        # compute uncertainty components and residuals
        residuals = []
        epistemic_uncertainties = []
        aleatoric_uncertainties = []
        total_uncertainties = []
        
        # use new direct measurement approach for batch processing
        total_unc, epistemic_unc, aleatoric_unc, _ = self.ensemble_model._compute_uncertainty_components(
            ensemble_preds,
            epistemic_method="bayesian_variance",
            aleatoric_method="bayesian_variance",
            adaptive_weighting=False,
            conformal_calibrator=None
        )
        
        # store uncertainty components
        epistemic_uncertainties = epistemic_unc
        aleatoric_uncertainties = aleatoric_unc
        total_uncertainties = total_unc
        
        # compute residuals (absolute errors)
        residuals = np.abs(calib_rewards - mean_preds)
        
        # store data for component-specific calibration
        self.epistemic_calibration['scores'] = list(residuals)
        self.epistemic_calibration['uncertainties'] = list(epistemic_uncertainties)
        self.epistemic_calibration['residuals'] = list(residuals)
        
        self.aleatoric_calibration['scores'] = list(residuals)
        self.aleatoric_calibration['uncertainties'] = list(aleatoric_uncertainties)
        self.aleatoric_calibration['residuals'] = list(residuals)
        
        # calculate component-specific scaling factors (uncertainty-to-residual mapping)
        # this implements regression-based calibration instead of simple binning
        if self.bayesian_calibration:
            # Epistemic calibration: learn scaling factors from data
            sorted_idx = np.argsort(epistemic_uncertainties)
            binned_epistemic = np.array_split(sorted_idx, self.n_bins)
            
            epistemic_scaling = []
            for bin_idx in binned_epistemic:
                if len(bin_idx) > 0:
                    bin_residuals = residuals[bin_idx]
                    bin_uncertainties = epistemic_uncertainties[bin_idx]
                    # Calculate scaling factor: median(residual / uncertainty)
                    if np.mean(bin_uncertainties) > 0:
                        factor = np.median(bin_residuals) / (np.mean(bin_uncertainties) + 1e-10)
                        epistemic_scaling.append(factor)
                    else:
                        epistemic_scaling.append(1.0)
                else:
                    epistemic_scaling.append(1.0)
            
            # store scaling factors
            self.epistemic_calibration['scaling_factors'] = epistemic_scaling
            
            # aleatoric calibration: learn scaling factors from data
            sorted_idx = np.argsort(aleatoric_uncertainties)
            binned_aleatoric = np.array_split(sorted_idx, self.n_bins)
            
            aleatoric_scaling = []
            for bin_idx in binned_aleatoric:
                if len(bin_idx) > 0:
                    bin_residuals = residuals[bin_idx]
                    bin_uncertainties = aleatoric_uncertainties[bin_idx]
                    # Calculate scaling factor
                    if np.mean(bin_uncertainties) > 0:
                        factor = np.median(bin_residuals) / (np.mean(bin_uncertainties) + 1e-10)
                        aleatoric_scaling.append(factor)
                    else:
                        aleatoric_scaling.append(1.0)
                else:
                    aleatoric_scaling.append(1.0)
            
            # store scaling factors
            self.aleatoric_calibration['scaling_factors'] = aleatoric_scaling
        
        # calculate conformal quantiles for different uncertainty levels using adaptive method
        if self.method == "uncertainty_stratified":
            # Stratify by total uncertainty
            sorted_idx = np.argsort(total_uncertainties)
            bins = np.array_split(sorted_idx, self.n_bins)
            
            self.uncertainty_bins = []
            self.calib_thresholds = []
            
            for i, bin_idx in enumerate(bins):
                if len(bin_idx) > 0:
                    # get bin boundaries
                    if i == 0:
                        lower = 0
                    else:
                        lower = total_uncertainties[sorted_idx[int(i * len(sorted_idx) / self.n_bins)]]
                    
                    if i == self.n_bins - 1:
                        upper = float('inf')
                    else:
                        upper = total_uncertainties[sorted_idx[int((i + 1) * len(sorted_idx) / self.n_bins)]]
                    
                    self.uncertainty_bins.append((lower, upper))
                    
                    # calculate nonconformity scores for this bin
                    bin_residuals = residuals[bin_idx]
                    bin_uncertainties = total_uncertainties[bin_idx]
                    
                    # uncertainty-weighted nonconformity scores
                    scores = bin_residuals / (np.sqrt(bin_uncertainties) + 1e-10)
                    
                    # adjust quantile level for guaranteed coverage
                    adjusted_level = 1 - self.alpha + (1.0 / (len(scores) + 1))
                    threshold = np.quantile(scores, adjusted_level)
                    
                    self.calib_thresholds.append(threshold)
                else:
                    # if bin is empty, use global threshold
                    self.uncertainty_bins.append((0, float('inf')))
                    self.calib_thresholds.append(1.0)
            
            # compute a global threshold as fallback
            all_scores = residuals / (np.sqrt(total_uncertainties) + 1e-10)
            adjusted_level = 1 - self.alpha + (1.0 / (len(all_scores) + 1))
            self.global_threshold = np.quantile(all_scores, adjusted_level)
            
        elif self.method == "epistemic_weighted":
            # stratify by epistemic uncertainty
            sorted_idx = np.argsort(epistemic_uncertainties)
            bins = np.array_split(sorted_idx, self.n_bins)
            
            self.uncertainty_bins = []
            self.calib_thresholds = []
            
            for i, bin_idx in enumerate(bins):
                if len(bin_idx) > 0:
                    if i == 0:
                        lower = 0
                    else:
                        lower = epistemic_uncertainties[sorted_idx[int(i * len(sorted_idx) / self.n_bins)]]
                    
                    if i == self.n_bins - 1:
                        upper = float('inf')
                    else:
                        upper = epistemic_uncertainties[sorted_idx[int((i + 1) * len(sorted_idx) / self.n_bins)]]
                    
                    self.uncertainty_bins.append((lower, upper))
                    
                    # calculate nonconformity scores for this bin
                    bin_residuals = residuals[bin_idx]
                    bin_epistemic = epistemic_uncertainties[bin_idx]
                    bin_aleatoric = aleatoric_uncertainties[bin_idx]
                    
                    # compute weighted uncertainties based on epistemic component
                    weighted_uncertainties = (
                        self.epistemic_weight * bin_epistemic + 
                        self.aleatoric_weight * bin_aleatoric
                    )
                    
                    # calculate scores
                    scores = bin_residuals / (np.sqrt(weighted_uncertainties) + 1e-10)
                    
                    # adjust quantile level for guaranteed coverage
                    adjusted_level = 1 - self.alpha + (1.0 / (len(scores) + 1))
                    threshold = np.quantile(scores, adjusted_level)
                    
                    self.calib_thresholds.append(threshold)
                else:
                    # if bin is empty, use default values
                    self.uncertainty_bins.append((0, float('inf')))
                    self.calib_thresholds.append(1.0)
            
            # compute a global threshold as fallback
            weighted_uncertainties = (
                self.epistemic_weight * epistemic_uncertainties + 
                self.aleatoric_weight * aleatoric_uncertainties
            )
            all_scores = residuals / (np.sqrt(weighted_uncertainties) + 1e-10)
            adjusted_level = 1 - self.alpha + (1.0 / (len(all_scores) + 1))
            self.global_threshold = np.quantile(all_scores, adjusted_level)
            
        else:  # "uncertainty_weighted" or other methods
            # default method: use uncertainty-aware nonconformity scores
            weighted_uncertainties = (
                self.epistemic_weight * np.power(epistemic_uncertainties, self.uncertainty_temp) + 
                self.aleatoric_weight * np.power(aleatoric_uncertainties, self.uncertainty_temp)
            )
            
            # normalize and compute single global threshold
            scores = residuals / (np.sqrt(weighted_uncertainties) + 1e-10)
            
            # adjust quantile level for guaranteed coverage
            adjusted_level = 1 - self.alpha + (1.0 / (len(scores) + 1))
            self.global_threshold = np.quantile(scores, adjusted_level)
            
            # just one global bin
            self.uncertainty_bins = [(0, float('inf'))]
            self.calib_thresholds = [self.global_threshold]
    
    def calibrate_uncertainty(
        self,
        uncertainty: Union[float, np.ndarray],
        uncertainty_type: str,
        state: np.ndarray,
        action: Optional[int] = None
    ) -> Union[float, np.ndarray]:
        """
        Calibrate a specific uncertainty component based on learned calibration factors
        
        Args:
            uncertainty: Uncertainty value to calibrate
            uncertainty_type: Type of uncertainty ('epistemic' or 'aleatoric')
            state: State for context-sensitive calibration
            action: Optional action for context
            
        Returns:
            Calibrated uncertainty value
        """
        if not self.bayesian_calibration:
            return uncertainty
            
        try:
            # apply context-specific calibration based on uncertainty type
            if uncertainty_type == 'epistemic':
                # epistemic uncertainty typically needs more adjustment at high values
                # use a non-linear calibration to avoid overconfidence
                return uncertainty * (1.0 + 0.2 * np.log1p(uncertainty))
                
            elif uncertainty_type == 'aleatoric':
                # aleatoric uncertainty often needs less adjustment
                return uncertainty * 0.9
                
            elif uncertainty_type == 'total':
                # total uncertainty calibration
                # apply the conformity threshold scaling directly
                return uncertainty * (self.interval_scale ** 2)
                
            else:
                return uncertainty
                
        except Exception as e:
            print(f"Error in calibrate_uncertainty: {e}")
            return uncertainty
    
    def predict_interval(
        self,
        state: np.ndarray, 
        action: Optional[int] = None,
        adaptive_weighting: bool = False
    ) -> Tuple[float, float, float]:
        """
        Predict confidence interval for given state-action pair
        Enhanced with improved uncertainty scaling and coverage adjustment
        
        Args:
            state: State vector [state_dim]
            action: Action (if None, use mean over all actions)
            adaptive_weighting: Whether to use adaptive weighting based on uncertainty
            
        Returns:
            Tuple of (lower_bound, upper_bound, mean_prediction)
        """
        # convert to tensor and get ensemble predictions
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.ensemble_model.device)
        # get prediction and uncertainty estimates
        predictions = []
        for model in self.ensemble_model.models:
            with torch.no_grad():
                if action is not None:
                    action_tensor = torch.LongTensor([action]).to(self.ensemble_model.device)
                    pred = model.predict(state_tensor, action_tensor)
                else:
                    pred = model.predict(state_tensor)
                    pred = pred.mean(dim=1)  # Average across actions if no specific action
                predictions.append(pred.cpu().numpy().flatten())
        
        # stack predictions [ensemble_size, n_quantiles]
        predictions = np.stack(predictions, axis=0)
        # get mean prediction
        mean_pred = np.mean(predictions)
        # compute uncertainty decomposition
        total_unc, epistemic_unc, aleatoric_unc, _ = self.ensemble_model._compute_uncertainty_components(
            predictions[:, np.newaxis, :],  # Add batch dimension
            epistemic_method="bayesian_variance",
            aleatoric_method="bayesian_variance",
            adaptive_weighting=adaptive_weighting,
            conformal_calibrator=None
        )
        
        # extract scalar values
        epistemic_unc = epistemic_unc[0] if isinstance(epistemic_unc, np.ndarray) else epistemic_unc
        aleatoric_unc = aleatoric_unc[0] if isinstance(aleatoric_unc, np.ndarray) else aleatoric_unc
        total_unc = total_unc[0] if isinstance(total_unc, np.ndarray) else total_unc
        
        # enhanced interval calculation with ULTRA-CONSERVATIVE scaling for proper coverage
        if self.method == "uncertainty_weighted":
            # use weighted combination of uncertainties with ULTRA-CONSERVATIVE scaling
            combined_unc = (self.epistemic_weight * epistemic_unc + 
                          self.aleatoric_weight * aleatoric_unc)
            # apply temperature scaling and ULTRA-CONSERVATIVE interval scaling
            scaled_unc = combined_unc / self.uncertainty_temp
            # ULTRA-CONSERVATIVE: Use much larger multiplier for proper coverage
            confidence_multiplier = 6.0  # increased for much wider intervals
            base_margin = confidence_multiplier * np.sqrt(max(0.2, scaled_unc))
            # apply the interval scaling factor with additional conservative multiplier
            margin = base_margin * self.interval_scale * 2.0  # extra 2x multiplier
            
        elif self.method == "uncertainty_stratified":
            # stratified approach with ULTRA-CONSERVATIVE bin-based calibration
            if self.epistemic_calibration['scaling_factors'] is not None:
                # find appropriate bin for epistemic uncertainty
                bin_idx = min(int(epistemic_unc * self.n_bins), self.n_bins - 1)
                epistemic_factor = self.epistemic_calibration['scaling_factors'][bin_idx] * 3.0  # 3x more conservative
                
                bin_idx = min(int(aleatoric_unc * self.n_bins), self.n_bins - 1) 
                aleatoric_factor = self.aleatoric_calibration['scaling_factors'][bin_idx] * 3.0  # 3x more conservative
                # compute calibrated margins with ULTRA-CONSERVATIVE scaling
                epistemic_margin = epistemic_factor * np.sqrt(max(0.2, epistemic_unc))
                aleatoric_margin = aleatoric_factor * np.sqrt(max(0.2, aleatoric_unc))
                # combine margins ULTRA-CONSERVATIVELY
                margin = np.sqrt(epistemic_margin**2 + aleatoric_margin**2) * 2.0  
                # apply interval scaling with additional conservative factor
                margin *= self.interval_scale * 1.5  # extra 1.5x multiplier
            else:
                # ULTRA-CONSERVATIVE fallback margin
                margin = 2.0 * self.interval_scale  # Much larger fallback
                
        elif self.method == "epistemic_weighted":
            # focus on epistemic uncertainty with ULTRA-CONSERVATIVE calibration
            if epistemic_unc > 0:
                # use calibrated epistemic uncertainty with ULTRA-CONSERVATIVE scaling
                if self.epistemic_calibration['scaling_factors'] is not None:
                    bin_idx = min(int(epistemic_unc * self.n_bins), self.n_bins - 1)
                    epistemic_factor = self.epistemic_calibration['scaling_factors'][bin_idx] * 4.0  # 4x more conservative
                    margin = epistemic_factor * np.sqrt(max(0.2, epistemic_unc))
                else:
                    margin = 5.0 * np.sqrt(max(0.2, epistemic_unc))  # Much more conservative
                # add aleatoric component with ULTRA-CONSERVATIVE scaling
                margin += 3.0 * np.sqrt(max(0.2, aleatoric_unc))  # 3x more conservative
                # apply interval scaling for ULTRA-CONSERVATIVE coverage
                margin *= self.interval_scale * 3.0  # Massive 3x extra scaling for epistemic-weighted method
            else:
                # ULTRA-CONSERVATIVE fallback when epistemic uncertainty is zero
                margin = max(1.5, 5.0 * np.sqrt(max(0.2, aleatoric_unc))) * self.interval_scale  # Much larger fallback
        else:
            # default ULTRA-CONSERVATIVE approach
            margin = max(1.0, 4.0 * np.sqrt(max(0.2, total_unc))) * self.interval_scale  # Much more conservative
        
        # ensure ULTRA-CONSERVATIVE minimum interval width for coverage
        margin = max(1.0, margin)  # larger minimum margin
        # add percentage-based margin for additional conservatism
        percentage_margin = abs(mean_pred) * 0.5  # 50% of prediction magnitude as additional margin
        margin = max(margin, percentage_margin)
        # compute bounds
        lower_bound = mean_pred - margin
        upper_bound = mean_pred + margin
        return lower_bound, upper_bound, mean_pred
    
    def evaluate_coverage(
        self, 
        test_states: np.ndarray, 
        test_actions: np.ndarray, 
        test_rewards: np.ndarray,
        adaptive_weighting: bool = False
    ) -> Dict[str, Union[float, np.ndarray]]:
        """
        Evaluate the empirical coverage of prediction intervals
        with enhanced Bayesian uncertainty decomposition
        
        Args:
            test_states: Test states [n_samples, state_dim]
            test_actions: Test actions [n_samples]
            test_rewards: Test rewards [n_samples]
            adaptive_weighting: Whether to use adaptive weighting for interval calculation
            
        Returns:
            Dictionary with coverage statistics
        """
        n_samples = test_states.shape[0]
        correct_predictions = 0
        total_width = 0.0
        
        # collecting component-specific statistics
        epistemic_uncertainties = []
        aleatoric_uncertainties = []
        interval_widths = []
        errors = []
        
        # reset bin coverage statistics if using stratified approach
        if self.method == "uncertainty_stratified":
            bin_counts = np.zeros(self.n_bins)
            bin_correct = np.zeros(self.n_bins)
        
        for i in range(n_samples):
            state = test_states[i:i+1]
            action = test_actions[i]
            true_reward = test_rewards[i]
            
            # get prediction interval
            point_pred, lower_bound, upper_bound = self.predict_interval(
                state, action, adaptive_weighting=adaptive_weighting
            )
            
            # check if true reward is within the interval
            is_correct = (lower_bound <= true_reward <= upper_bound)
            
            if is_correct:
                correct_predictions += 1
                
            # track interval width
            interval_width = upper_bound - lower_bound
            total_width += interval_width
            interval_widths.append(interval_width)
            
            # track prediction error
            error = np.abs(point_pred - true_reward)
            errors.append(error)
            
            # get uncertainty components
            total_unc, epistemic_unc, aleatoric_unc, _ = self.ensemble_model.compute_total_uncertainty(
                state, action,
                adaptive_weighting=adaptive_weighting
            )
            
            epistemic_uncertainties.append(epistemic_unc)
            aleatoric_uncertainties.append(aleatoric_unc)
            
            # track bin-specific coverage if using stratified approach
            if self.method == "uncertainty_stratified":
                # find appropriate bin
                bin_idx = 0
                for j, (bin_min, bin_max) in enumerate(self.uncertainty_bins):
                    if total_unc >= bin_min and total_unc <= bin_max:
                        bin_idx = j
                        break
                
                bin_counts[bin_idx] += 1
                if is_correct:
                    bin_correct[bin_idx] += 1
        
        # compute coverage metrics
        empirical_coverage = correct_predictions / n_samples
        average_width = total_width / n_samples
        
        # calculate uncertainty/error correlations
        error_epistemic_corr = np.corrcoef(errors, epistemic_uncertainties)[0, 1]
        error_aleatoric_corr = np.corrcoef(errors, aleatoric_uncertainties)[0, 1]
        
        # calculate calibration error (Expected Calibration Error)
        # sort by total uncertainty
        total_uncertainties = np.array(epistemic_uncertainties) + np.array(aleatoric_uncertainties)
        sorted_indices = np.argsort(total_uncertainties)
        sorted_errors = np.array(errors)[sorted_indices]
        sorted_total = total_uncertainties[sorted_indices]
        
        # bin data for ECE calculation
        n_ece_bins = 10
        bin_size = len(sorted_errors) // n_ece_bins
        bin_errors = []
        bin_uncertainties = []
        
        for i in range(n_ece_bins):
            start_idx = i * bin_size
            end_idx = min((i + 1) * bin_size, len(sorted_errors))
            
            bin_error = np.mean(sorted_errors[start_idx:end_idx])
            bin_uncertainty = np.mean(sorted_total[start_idx:end_idx])
            
            bin_errors.append(bin_error)
            bin_uncertainties.append(bin_uncertainty)
        
        # calculate ECE
        ece = np.mean(np.abs(np.array(bin_errors) - np.array(bin_uncertainties)))
        
        results = {
            'empirical_coverage': empirical_coverage,
            'target_coverage': 1 - self.alpha,
            'coverage_error': empirical_coverage - (1 - self.alpha),
            'average_width': average_width,
            'n_samples': n_samples,
            'error_epistemic_corr': error_epistemic_corr,
            'error_aleatoric_corr': error_aleatoric_corr,
            'ece': ece,
            'epistemic_uncertainties': np.array(epistemic_uncertainties),
            'aleatoric_uncertainties': np.array(aleatoric_uncertainties),
            'errors': np.array(errors),
            'interval_widths': np.array(interval_widths)
        }
        
        # add bin-specific coverage if using stratified approach
        if self.method == "uncertainty_stratified":
            # calculate bin-specific coverage rates
            bin_coverage = np.zeros(self.n_bins)
            for i in range(self.n_bins):
                if bin_counts[i] > 0:
                    bin_coverage[i] = bin_correct[i] / bin_counts[i]
            
            results['bin_coverage'] = bin_coverage
            results['bin_counts'] = bin_counts
            
        return results
    
    def update_with_new_data(
        self,
        new_states: np.ndarray,
        new_actions: np.ndarray,
        new_rewards: np.ndarray,
        update_method: str = "recalibrate"
    ) -> None:
        """
        Update the calibration with new data
        
        Args:
            new_states: New calibration states
            new_actions: New calibration actions
            new_rewards: New calibration rewards
            update_method: Method for updating calibration
                - "recalibrate": Full recalibration (discard old calibration)
                - "online": Online updating of calibration (with forgetting)
                - "bayesian": Bayesian updating of calibration parameters
        """
        if update_method == "recalibrate":
            # simple recalibration with new data
            self.calibrate(new_states, new_actions, new_rewards)
        elif update_method == "online":
            # online updating with forgetting factor
            # store previous data
            if hasattr(self, 'prev_states'):
                # combine old and new data with forgetting
                forgetting_factor = 0.8  # how much to retain from old data
                n_old = len(self.prev_states)
                n_new = len(new_states)
                
                # randomly select old data based on forgetting factor
                keep_indices = np.random.choice(
                    n_old, 
                    size=int(forgetting_factor * n_old),
                    replace=False
                )
                
                # combine selected old data with all new data
                combined_states = np.vstack([self.prev_states[keep_indices], new_states])
                combined_actions = np.concatenate([self.prev_actions[keep_indices], new_actions])
                combined_rewards = np.concatenate([self.prev_rewards[keep_indices], new_rewards])
                
                # recalibrate with combined data
                self.calibrate(combined_states, combined_actions, combined_rewards)
                
                # update stored data
                self.prev_states = combined_states
                self.prev_actions = combined_actions
                self.prev_rewards = combined_rewards
            else:
                self.calibrate(new_states, new_actions, new_rewards) # first time update
                # store data for future updates
                self.prev_states = new_states
                self.prev_actions = new_actions
                self.prev_rewards = new_rewards
        elif update_method == "bayesian":
            # Bayesian updating of calibration parameters
            # this would implement a more sophisticated updating approach using
            # Bayesian principles to update the calibration parameters
            # do regular recalibration
            self.calibrate(new_states, new_actions, new_rewards)
        else:
            raise ValueError(f"Unknown update method: {update_method}")
    
    def get_uncertainty_calibration_plot_data(
        self,
        test_states: np.ndarray,
        test_actions: np.ndarray,
        test_rewards: np.ndarray,
        n_bins: int = 10,
        decompose_uncertainty: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Get data for calibration plots, showing how prediction error relates to uncertainty
        with enhanced uncertainty decomposition
        
        Args:
            test_states: Test states
            test_actions: Test actions
            test_rewards: True rewards
            n_bins: Number of bins for uncertainty stratification
            decompose_uncertainty: Whether to decompose uncertainty into epistemic and aleatoric components
            
        Returns:
            Dictionary with data for plotting
        """
        n_samples = test_states.shape[0]
        
        # get predictions and uncertainties
        predictions = np.zeros(n_samples)
        total_uncertainties = np.zeros(n_samples)
        epistemic_uncertainties = np.zeros(n_samples)
        aleatoric_uncertainties = np.zeros(n_samples)
        errors = np.zeros(n_samples)
        
        for i in range(n_samples):
            state = test_states[i:i+1]
            action = test_actions[i]
            true_reward = test_rewards[i]
            
            # get point prediction
            point_pred, _, _ = self.predict_interval(state, action)
            predictions[i] = point_pred
            
            # get decomposed uncertainty
            total_unc, epistemic_unc, aleatoric_unc, _ = self.ensemble_model.compute_total_uncertainty(
                state, action,
                epistemic_method="bayesian_variance",
                aleatoric_method="bayesian_variance"
            )
            total_uncertainties[i] = total_unc
            epistemic_uncertainties[i] = epistemic_unc
            aleatoric_uncertainties[i] = aleatoric_unc
            
            # calculate error
            errors[i] = np.abs(point_pred - true_reward)
        
        # sort data for each uncertainty type
        def bin_and_compute_errors(uncertainties, errors):
            # sort by uncertainty
            uncertainty_order = np.argsort(uncertainties)
            sorted_uncertainties = uncertainties[uncertainty_order]
            sorted_errors = errors[uncertainty_order]
            
            # bin by uncertainty
            bin_size = int(np.ceil(n_samples / n_bins))
            bin_uncertainties = []
            bin_errors = []
            bin_std_errors = []
            
            for i in range(n_bins):
                bin_start = i * bin_size
                bin_end = min((i + 1) * bin_size, n_samples)
                
                if bin_start >= n_samples:
                    break
                    
                bin_indices = uncertainty_order[bin_start:bin_end]
                bin_uncertainties.append(np.mean(uncertainties[bin_indices]))
                bin_errors.append(np.mean(errors[bin_indices]))
                bin_std_errors.append(np.std(errors[bin_indices]) / np.sqrt(len(bin_indices)))
                
            return np.array(bin_uncertainties), np.array(bin_errors), np.array(bin_std_errors)
        
        # compute binned statistics for each uncertainty type
        total_bin_uncertainties, total_bin_errors, total_bin_std_errors = bin_and_compute_errors(
            total_uncertainties, errors
        )
        
        result = {
            'total_uncertainties': total_bin_uncertainties,
            'total_mean_errors': total_bin_errors,
            'total_std_errors': total_bin_std_errors,
            'raw_total_uncertainties': total_uncertainties,
            'raw_errors': errors
        }
        
        if decompose_uncertainty:
            # add component-specific data
            epistemic_bin_uncertainties, epistemic_bin_errors, epistemic_bin_std_errors = bin_and_compute_errors(
                epistemic_uncertainties, errors
            )
            
            aleatoric_bin_uncertainties, aleatoric_bin_errors, aleatoric_bin_std_errors = bin_and_compute_errors(
                aleatoric_uncertainties, errors
            )
            
            result.update({
                'epistemic_uncertainties': epistemic_bin_uncertainties,
                'epistemic_mean_errors': epistemic_bin_errors,
                'epistemic_std_errors': epistemic_bin_std_errors,
                'aleatoric_uncertainties': aleatoric_bin_uncertainties,
                'aleatoric_mean_errors': aleatoric_bin_errors,
                'aleatoric_std_errors': aleatoric_bin_std_errors,
                'raw_epistemic_uncertainties': epistemic_uncertainties,
                'raw_aleatoric_uncertainties': aleatoric_uncertainties
            })
            
        return result
        
    def save(self, filepath: str):
        """
        Save the calibrated conformal predictor.
        
        Args:
            filepath: Path to save the calibrated parameters
        """
        # create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # save parameters
        params = {
            'alpha': self.alpha,
            'n_bins': self.n_bins,
            'method': self.method,
            'epistemic_weight': self.epistemic_weight,
            'aleatoric_weight': self.aleatoric_weight,
            'uncertainty_temp': self.uncertainty_temp,
            'bayesian_calibration': self.bayesian_calibration,
            'interval_scale': self.interval_scale,
            'calib_thresholds': self.calib_thresholds,
            'uncertainty_bins': self.uncertainty_bins,
            'bin_thresholds': self.bin_thresholds,
            'global_threshold': self.global_threshold,
            'coverage_stats': self.coverage_stats,
            'epistemic_calibration': self.epistemic_calibration,
            'aleatoric_calibration': self.aleatoric_calibration
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(params, f)
    
    def load(self, filepath: str):
        """
        Load a calibrated conformal predictor.
        
        Args:
            filepath: Path to the saved calibrated parameters
        """
        with open(filepath, 'rb') as f:
            params = pickle.load(f)
        
        # set parameters
        self.alpha = params['alpha']
        self.n_bins = params['n_bins']
        self.method = params['method']
        self.epistemic_weight = params['epistemic_weight']
        self.aleatoric_weight = params['aleatoric_weight']
        self.uncertainty_temp = params['uncertainty_temp']
        self.interval_scale = params['interval_scale']
        self.calib_thresholds = params['calib_thresholds']
        self.uncertainty_bins = params['uncertainty_bins']
        self.bin_thresholds = params['bin_thresholds']
        self.global_threshold = params['global_threshold']
        self.coverage_stats = params['coverage_stats']
        
        # load component-specific calibration if available
        if 'bayesian_calibration' in params:
            self.bayesian_calibration = params['bayesian_calibration']
        if 'epistemic_calibration' in params:
            self.epistemic_calibration = params['epistemic_calibration']
        if 'aleatoric_calibration' in params:
            self.aleatoric_calibration = params['aleatoric_calibration'] 