import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.signal import convolve
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class ECGGaussianFitter:
    """
    Fits ECG components (P, Q, R, S, T) to double Gaussian functions.
    """
    
    # Component duration and sample count defaults (at 360 Hz sampling rate)
    COMPONENT_PARAMS = {
        'P': {'duration_ms': 90, 'samples': 32},  # 80-100ms
        'Q': {'duration_ms': 40, 'samples': 14},  # Part of QRS
        'R': {'duration_ms': 90, 'samples': 32},  # 80-100ms (QRS)
        'S': {'duration_ms': 40, 'samples': 14},  # Part of QRS
        'T': {'duration_ms': 140, 'samples': 50}  # 120-160ms
    }
    
    def __init__(self, sampling_rate: int = 360):
        """
        Initialize the ECG Gaussian Fitter.
        
        Args:
            sampling_rate: Sampling rate of ECG signal in Hz
        """
        self.fs = sampling_rate
        self._update_component_samples()
    
    def _update_component_samples(self):
        """Update sample counts based on sampling rate."""
        for comp in self.COMPONENT_PARAMS:
            duration_s = self.COMPONENT_PARAMS[comp]['duration_ms'] / 1000
            self.COMPONENT_PARAMS[comp]['samples'] = int(duration_s * self.fs)
    
    def _extract_segments_from_fiducials(self, 
                                       full_beat_segment: np.ndarray,
                                       fiducial_points: Dict[str, int]) -> Dict[str, np.ndarray]:
        """
        Helper method to extract PQRST segments from fiducial points.
        This is used as a fallback for backward compatibility.
        
        Parameters:
        -----------
        full_beat_segment : np.ndarray
            Complete beat segment
        fiducial_points : Dict[str, int]
            Dictionary mapping wave names to their indices
            
        Returns:
        --------
        Dict[str, np.ndarray]
            Dictionary containing P, Q, R, S, T wave segments
        """
        segments = {}
        
        # Define extraction windows around detected waves (in samples)
        window_sizes = {
            'P': int(0.12 * self.fs),  # 120ms window for P wave
            'Q': int(0.06 * self.fs),  # 60ms window for Q wave  
            'R': int(0.08 * self.fs),  # 80ms window for R wave
            'S': int(0.06 * self.fs),  # 60ms window for S wave
            'T': int(0.16 * self.fs),  # 160ms window for T wave
        }
        
        for wave_name in ['P', 'Q', 'R', 'S', 'T']:
            if wave_name in fiducial_points:
                wave_idx = fiducial_points[wave_name]
                half_window = window_sizes[wave_name] // 2
                
                start_idx = max(0, wave_idx - half_window)
                end_idx = min(len(full_beat_segment), wave_idx + half_window)
                
                if end_idx > start_idx:
                    segments[wave_name] = full_beat_segment[start_idx:end_idx]
                else:
                    segments[wave_name] = np.array([])
            else:
                segments[wave_name] = np.array([])
                
        return segments
    
    def calculate_sigma_bounds(self, component: str, x_R: float = None) -> Tuple[float, float]:
        """
        Calculate sigma_min and sigma_max for a component.
        
        Args:
            component: Component name ('P', 'Q', 'R', 'S', 'T')
            x_R: Total time duration of component. If None, uses default.
        
        Returns:
            (sigma_min, sigma_max) tuple
        """
        Ns = self.COMPONENT_PARAMS[component]['samples']
        
        if x_R is None:
            x_R = self.COMPONENT_PARAMS[component]['duration_ms'] / 1000
        
        sigma_min = x_R / (Ns * 5)
        sigma_max = x_R / 3
        
        return sigma_min, sigma_max
    
    def create_gaussian_filter(self, Ns: int, sigma: float) -> np.ndarray:
        """
        Create Gaussian filter (Equation 4 and 5).
        
        Args:
            Ns: Number of samples
            sigma: Sigma value for Gaussian
        
        Returns:
            Gaussian filter array
        """
        # S = -ceil(Ns/2)+1 : 1 : floor(ceil(Ns/2))
        start = -(np.ceil(Ns / 2).astype(int) + 1)
        end = np.floor(np.ceil(Ns / 2)).astype(int)
        S = np.arange(start, end + 1, 1)
        
        # Gfilt = exp(-(S^2 / sigma^2))
        G_filt = np.exp(-(S**2) / (sigma**2))
        
        return G_filt
    
    def matched_filter_search(self, ecg_segment: np.ndarray, component: str, 
                             x_R: float = None) -> Dict:
        """
        Step 1.1-1.5: Search for optimal first Gaussian parameters using matched filtering.
        
        Args:
            ecg_segment: ECG signal segment for the component
            component: Component name ('P', 'Q', 'R', 'S', 'T')
            x_R: Total time duration of component
        
        Returns:
            Dictionary with optimal parameters {A1, sigma1, t1, rmse}
        """
        Ns = len(ecg_segment)
        sigma_min, sigma_max = self.calculate_sigma_bounds(component, x_R)
        
        # Step 1.2: Increment sigma from sigma_min to sigma_max
        sigma_values = np.arange(sigma_min, sigma_max + 0.3, 0.3)
        
        best_rmse = np.inf
        best_params = {}
        
        # Steps 1.2-1.4: Iterate through all sigma values
        for sigma1 in sigma_values:
            # Create Gaussian filter
            G_filt = self.create_gaussian_filter(Ns, sigma1)
            
            # Convolve ECG with Gaussian filter (matched filtering)
            response = convolve(ecg_segment, G_filt, mode='same')
            
            # Find maximum response position and amplitude
            max_idx = np.argmax(np.abs(response))
            t1 = max_idx  # Position in samples
            A1 = ecg_segment[max_idx]  # Amplitude at peak
            
            # Generate single Gaussian with these parameters
            t = np.arange(len(ecg_segment))
            gaussian_model = A1 * np.exp(-((t - t1)**2) / (sigma1**2))
            
            # Calculate RMSE (Step 1.3)
            rmse = np.sqrt(np.mean((gaussian_model - ecg_segment)**2))
            
            # Step 1.3: Store if best
            if rmse < best_rmse:
                best_rmse = rmse
                best_params = {
                    'A1': A1,
                    'sigma1': sigma1,
                    't1': t1,
                    'rmse': rmse
                }
        
        # Step 1.6: Approximate second Gaussian parameters
        best_params['A2'] = best_params['A1']
        best_params['sigma2'] = best_params['sigma1']
        best_params['t2'] = best_params['t1']
        
        return best_params
    
    def double_gaussian_model(self, t: np.ndarray, params: List[float]) -> np.ndarray:
        """
        Generate double Gaussian model for ECG component.
        
        Args:
            t: Time vector
            params: [A1, t1, sigma1, A2, t2, sigma2, C]
        
        Returns:
            ECG model signal
        """
        A1, t1, sigma1, A2, t2, sigma2, C = params
        
        gaussian1 = A1 * np.exp(-((t - t1)**2) / (sigma1**2))
        gaussian2 = A2 * np.exp(-((t - t2)**2) / (sigma2**2))
        
        return gaussian1 + gaussian2 + C
    
    def objective_function(self, params: np.ndarray, t: np.ndarray, 
                          ecg_real: np.ndarray) -> float:
        """
        Objective function for optimization (RMSE).
        
        Args:
            params: [A1, t1, sigma1, A2, t2, sigma2, C]
            t: Time vector
            ecg_real: Real ECG signal
        
        Returns:
            RMSE value
        """
        ecg_model = self.double_gaussian_model(t, params)
        rmse = np.sqrt(np.mean((ecg_model - ecg_real)**2))
        return rmse
    
    def optimize_with_global_search(self, ecg_segment: np.ndarray, 
                                    initial_params: Dict,
                                    method: str = 'differential_evolution') -> Dict:
        """
        Step 2: Global optimization to refine Gaussian parameters.
        
        Args:
            ecg_segment: ECG signal segment
            initial_params: Initial parameters from matched filtering
            method: 'differential_evolution' or 'multistart'
        
        Returns:
            Optimized parameters dictionary
        """
        t = np.arange(len(ecg_segment))
        
        # Initial guess from matched filtering
        x0 = [
            initial_params['A1'],
            initial_params['t1'],
            initial_params['sigma1'],
            initial_params['A2'],
            initial_params['t2'],
            initial_params['sigma2'],
            np.mean(ecg_segment)  # Baseline offset
        ]
        
        # Define bounds for optimization
        A_max = np.max(np.abs(ecg_segment)) * 2
        bounds = [
            (-A_max, A_max),  # A1
            (0, len(ecg_segment) - 1),  # t1
            (0.1, len(ecg_segment) / 2),  # sigma1
            (-A_max, A_max),  # A2
            (0, len(ecg_segment) - 1),  # t2
            (0.1, len(ecg_segment) / 2),  # sigma2
            (-A_max / 2, A_max / 2)  # C (baseline)
        ]
        
        if method == 'differential_evolution':
            # Global optimization using differential evolution
            result = differential_evolution(
                self.objective_function,
                bounds=bounds,
                args=(t, ecg_segment),
                seed=42,
                maxiter=1000,
                popsize=15,
                atol=1e-6,
                tol=1e-6
            )
            optimal_params = result.x
            final_rmse = result.fun
            
        else:  # multistart
            # Multi-start local optimization
            best_result = None
            best_rmse = np.inf
            
            # Try multiple random starts
            n_starts = 10
            for i in range(n_starts):
                if i == 0:
                    # First start with matched filter result
                    start_params = x0
                else:
                    # Random starts within bounds
                    start_params = [np.random.uniform(b[0], b[1]) for b in bounds]
                
                result = minimize(
                    self.objective_function,
                    start_params,
                    args=(t, ecg_segment),
                    method='L-BFGS-B',
                    bounds=bounds
                )
                
                if result.fun < best_rmse:
                    best_rmse = result.fun
                    best_result = result
            
            optimal_params = best_result.x
            final_rmse = best_result.fun
        
        # Package results
        optimized = {
            'A1': optimal_params[0],
            't1': optimal_params[1],
            'sigma1': optimal_params[2],
            'A2': optimal_params[3],
            't2': optimal_params[4],
            'sigma2': optimal_params[5],
            'C': optimal_params[6],
            'rmse': final_rmse
        }
        
        return optimized
    
    def fit_component(self, ecg_segment: np.ndarray, component: str,
                     use_global_optimization: bool = True,
                     optimization_method: str = 'differential_evolution') -> Dict:
        """
        Complete fitting pipeline for a single ECG component.
        
        Args:
            ecg_segment: ECG signal segment for the component
            component: Component name ('P', 'Q', 'R', 'S', 'T')
            use_global_optimization: Whether to apply global optimization (Step 2)
            optimization_method: 'differential_evolution' or 'multistart'
        
        Returns:
            Dictionary with final parameters and RMSE
        """
        # Step 1: Matched filtering approximation
        initial_params = self.matched_filter_search(ecg_segment, component)
        
        if not use_global_optimization:
            # Add baseline offset
            initial_params['C'] = 0.0
            return initial_params
        
        # Step 2: Global optimization
        optimized_params = self.optimize_with_global_search(
            ecg_segment,
            initial_params,
            method=optimization_method
        )
        
        return optimized_params
    
    def fit_all_components(self, segmented_ecg: Dict[str, np.ndarray],
                          use_global_optimization: bool = True,
                          optimization_method: str = 'differential_evolution',
                          epsilon: float = 0.01) -> Dict[str, Dict]:
        """
        Fit all ECG components (P, Q, R, S, T) to double Gaussian functions.
        
        Args:
            segmented_ecg: Dictionary with keys 'P', 'Q', 'R', 'S', 'T' 
                          and values as ECG signal arrays
            use_global_optimization: Whether to apply global optimization
            optimization_method: 'differential_evolution' or 'multistart'
            epsilon: RMSE threshold for acceptable fit
        
        Returns:
            Dictionary of parameters for each component
        """
        all_params = {}
        
        for component in ['P', 'Q', 'R', 'S', 'T']:
            if component not in segmented_ecg:
                print(f"Warning: Component {component} not found in segmented_ecg")
                continue
            
            ecg_segment = segmented_ecg[component]
            
            print(f"Fitting component {component}...")
            params = self.fit_component(
                ecg_segment,
                component,
                use_global_optimization=use_global_optimization,
                optimization_method=optimization_method
            )
            
            all_params[component] = params
            
            # Check if RMSE is below threshold
            if params['rmse'] < epsilon:
                print(f"  ✓ {component}: RMSE = {params['rmse']:.6f} < ε ({epsilon})")
            else:
                print(f"  ⚠ {component}: RMSE = {params['rmse']:.6f} >= ε ({epsilon})")
        
        return all_params
    
    def reconstruct_ecg(self, params: Dict[str, Dict], 
                       segment_lengths: Dict[str, int]) -> np.ndarray:
        """
        Reconstruct complete ECG beat from Gaussian parameters.
        
        Args:
            params: Dictionary of parameters for each component
            segment_lengths: Dictionary with length of each component segment
        
        Returns:
            Reconstructed ECG signal
        """
        reconstructed_segments = []
        
        for component in ['P', 'Q', 'R', 'S', 'T']:
            if component not in params:
                continue
            
            length = segment_lengths[component]
            t = np.arange(length)
            
            p = params[component]
            param_list = [p['A1'], p['t1'], p['sigma1'], 
                         p['A2'], p['t2'], p['sigma2'], p['C']]
            
            segment_reconstructed = self.double_gaussian_model(t, param_list)
            reconstructed_segments.append(segment_reconstructed)
        
        return np.concatenate(reconstructed_segments)
    
    def reconstruct_component(self, length: int, component_params: Dict) -> np.ndarray:
        """
        Reconstruct a single ECG component from Gaussian parameters.
        
        Args:
            length: Length of the component segment
            component_params: Dictionary with Gaussian parameters
        
        Returns:
            Reconstructed component signal
        """
        t = np.arange(length)
        param_list = [
            component_params['A1'], 
            component_params['t1'], 
            component_params['sigma1'],
            component_params['A2'], 
            component_params['t2'], 
            component_params['sigma2'], 
            component_params['C']
        ]
        return self.double_gaussian_model(t, param_list)
    
    def plot_individual_components(self, segmented_ecg: Dict[str, np.ndarray],
                                   params: Dict[str, Dict],
                                   figsize: Tuple[int, int] = (15, 10)):
        """
        Plot individual ECG components with their Gaussian fits.
        
        Args:
            segmented_ecg: Dictionary of real ECG segments
            params: Dictionary of fitted parameters
            figsize: Figure size tuple
        """
        components = ['P', 'Q', 'R', 'S', 'T']
        colors = ['blue', 'green', 'red', 'orange', 'purple']
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
        
        for idx, (component, color) in enumerate(zip(components, colors)):
            if component not in segmented_ecg or component not in params:
                axes[idx].text(0.5, 0.5, f'{component} not available',
                             ha='center', va='center', fontsize=12)
                axes[idx].set_title(f'{component} Wave')
                continue
            
            # Get real signal
            real_signal = segmented_ecg[component]
            t = np.arange(len(real_signal))
            
            # Get model signal
            p = params[component]
            param_list = [p['A1'], p['t1'], p['sigma1'], 
                         p['A2'], p['t2'], p['sigma2'], p['C']]
            model_signal = self.double_gaussian_model(t, param_list)
            
            # Plot
            axes[idx].plot(t, real_signal, 'o-', color=color, alpha=0.6, 
                          label='Real', linewidth=2, markersize=4)
            axes[idx].plot(t, model_signal, '--', color='black', 
                          label='Model', linewidth=2)
            
            # Add individual Gaussians
            gaussian1 = p['A1'] * np.exp(-((t - p['t1'])**2) / (p['sigma1']**2))
            gaussian2 = p['A2'] * np.exp(-((t - p['t2'])**2) / (p['sigma2']**2))
            axes[idx].plot(t, gaussian1 + p['C'], ':', color='gray', 
                          alpha=0.5, label='G1', linewidth=1.5)
            axes[idx].plot(t, gaussian2 + p['C'], ':', color='darkgray', 
                          alpha=0.5, label='G2', linewidth=1.5)
            
            # Add RMSE text
            rmse = p['rmse']
            axes[idx].text(0.05, 0.95, f'RMSE: {rmse:.6f}', 
                          transform=axes[idx].transAxes,
                          verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                          fontsize=10, fontweight='bold')
            
            axes[idx].set_title(f'{component} Wave Component', fontweight='bold', fontsize=12)
            axes[idx].set_xlabel('Sample Index', fontsize=10)
            axes[idx].set_ylabel('Amplitude', fontsize=10)
            axes[idx].legend(loc='upper right', fontsize=8)
            axes[idx].grid(True, alpha=0.3)
        
        # Remove extra subplot
        axes[-1].axis('off')
        
        plt.tight_layout()
        plt.savefig('individual_components.png', dpi=150, bbox_inches='tight')
        print("✓ Individual component plots saved as 'individual_components.png'")
        plt.show()
    
    def plot_complete_beat(self, segmented_ecg: Dict[str, np.ndarray],
                          params: Dict[str, Dict],
                          figsize: Tuple[int, int] = (16, 8)):
        """
        Plot complete ECG beat with model overlay.
        
        Args:
            segmented_ecg: Dictionary of real ECG segments
            params: Dictionary of fitted parameters
            figsize: Figure size tuple
        """
        # Concatenate real signals
        components = ['P', 'Q', 'R', 'S', 'T']
        real_segments = []
        model_segments = []
        segment_boundaries = [0]
        component_labels = []
        
        for component in components:
            if component not in segmented_ecg or component not in params:
                continue
            
            real_signal = segmented_ecg[component]
            t = np.arange(len(real_signal))
            
            # Get model signal
            p = params[component]
            param_list = [p['A1'], p['t1'], p['sigma1'], 
                         p['A2'], p['t2'], p['sigma2'], p['C']]
            model_signal = self.double_gaussian_model(t, param_list)
            
            real_segments.append(real_signal)
            model_segments.append(model_signal)
            segment_boundaries.append(segment_boundaries[-1] + len(real_signal))
            component_labels.append((segment_boundaries[-2] + len(real_signal)//2, component))
        
        real_complete = np.concatenate(real_segments)
        model_complete = np.concatenate(model_segments)
        t_complete = np.arange(len(real_complete))
        
        # Calculate overall RMSE
        overall_rmse = np.sqrt(np.mean((model_complete - real_complete)**2))
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, 
                                        gridspec_kw={'height_ratios': [3, 1]})
        
        # Main plot - Complete ECG beat
        ax1.plot(t_complete, real_complete, 'o-', color='blue', alpha=0.6,
                label='Real ECG', linewidth=2.5, markersize=5)
        ax1.plot(t_complete, model_complete, '--', color='red',
                label='Model (Double Gaussians)', linewidth=2.5)
        
        # Add segment boundaries
        for boundary in segment_boundaries[1:-1]:
            ax1.axvline(x=boundary, color='gray', linestyle=':', alpha=0.5, linewidth=1.5)
        
        # Add component labels
        for pos, label in component_labels:
            ax1.text(pos, ax1.get_ylim()[1] * 0.9, label,
                    ha='center', va='top', fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
        
        # Add RMSE text box
        textstr = f'Overall RMSE: {overall_rmse:.6f}'
        ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes,
                verticalalignment='top', fontsize=14, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9, pad=1))
        
        ax1.set_title('Complete ECG Beat: Real vs Model', fontsize=16, fontweight='bold', pad=20)
        ax1.set_xlabel('Sample Index', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Amplitude', fontsize=12, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=11, framealpha=0.9)
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        # Residual plot
        residuals = real_complete - model_complete
        ax2.plot(t_complete, residuals, color='darkred', linewidth=1.5, alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax2.fill_between(t_complete, residuals, alpha=0.3, color='red')
        
        # Add segment boundaries to residual plot
        for boundary in segment_boundaries[1:-1]:
            ax2.axvline(x=boundary, color='gray', linestyle=':', alpha=0.5, linewidth=1.5)
        
        ax2.set_title('Residuals (Real - Model)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Sample Index', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Error', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig('complete_ecg_beat.png', dpi=150, bbox_inches='tight')
        print("✓ Complete ECG beat plot saved as 'complete_ecg_beat.png'")
        plt.show()
        
        return overall_rmse
    
    def plot_summary_table(self, params: Dict[str, Dict], figsize: Tuple[int, int] = (12, 6)):
        """
        Create a summary table of all fitted parameters.
        
        Args:
            params: Dictionary of fitted parameters for all components
            figsize: Figure size tuple
        """
        fig, ax = plt.subplots(figsize=figsize)
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare table data
        headers = ['Component', 'A₁', 't₁', 'σ₁', 'A₂', 't₂', 'σ₂', 'C', 'RMSE']
        table_data = []
        
        for component in ['P', 'Q', 'R', 'S', 'T']:
            if component not in params:
                continue
            p = params[component]
            row = [
                component,
                f"{p['A1']:.4f}",
                f"{p['t1']:.2f}",
                f"{p['sigma1']:.4f}",
                f"{p['A2']:.4f}",
                f"{p['t2']:.2f}",
                f"{p['sigma2']:.4f}",
                f"{p['C']:.4f}",
                f"{p['rmse']:.6f}"
            ]
            table_data.append(row)
        
        # Create table
        table = ax.table(cellText=table_data, colLabels=headers,
                        cellLoc='center', loc='center',
                        colWidths=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.12])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.5)
        
        # Style header
        for i in range(len(headers)):
            cell = table[(0, i)]
            cell.set_facecolor('#4CAF50')
            cell.set_text_props(weight='bold', color='white', fontsize=11)
        
        # Style cells
        colors = ['#E3F2FD', '#BBDEFB']
        for i, row in enumerate(table_data):
            for j in range(len(headers)):
                cell = table[(i+1, j)]
                cell.set_facecolor(colors[i % 2])
                if j == 0:  # Component column
                    cell.set_text_props(weight='bold')
                if j == len(headers) - 1:  # RMSE column
                    cell.set_text_props(weight='bold', color='darkred')
        
        plt.title('ECG Gaussian Fitting Parameters Summary', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.savefig('parameters_summary.png', dpi=150, bbox_inches='tight')
        print("✓ Parameters summary table saved as 'parameters_summary.png'")
        plt.show()


# Example usage function
def example_usage():
    """
    Example of how to use the ECG Gaussian Fitter with visualization.
    """
    # Create synthetic ECG segments (replace with your real data)
    np.random.seed(42)
    
    # Simulated segmented ECG data with realistic shapes
    t_p = np.linspace(0, 1, 32)
    P = 0.2 * np.exp(-((t_p - 0.5)**2) / 0.05) + np.random.randn(32) * 0.02
    
    t_q = np.linspace(0, 1, 14)
    Q = -0.1 * np.exp(-((t_q - 0.5)**2) / 0.03) + np.random.randn(14) * 0.01
    
    t_r = np.linspace(0, 1, 32)
    R = 1.2 * np.exp(-((t_r - 0.5)**2) / 0.02) + np.random.randn(32) * 0.03
    
    t_s = np.linspace(0, 1, 14)
    S = -0.2 * np.exp(-((t_s - 0.5)**2) / 0.03) + np.random.randn(14) * 0.015
    
    t_t = np.linspace(0, 1, 50)
    T = 0.3 * np.exp(-((t_t - 0.5)**2) / 0.08) + np.random.randn(50) * 0.02
    
    segmented_ecg = {
        'P': P,
        'Q': Q,
        'R': R,
        'S': S,
        'T': T
    }
    
    # Initialize fitter
    print("="*70)
    print("ECG GAUSSIAN FITTING - Starting Analysis")
    print("="*70)
    fitter = ECGGaussianFitter(sampling_rate=360)
    
    # Fit all components
    print("\n" + "-"*70)
    print("STEP 1 & 2: Matched Filtering + Global Optimization")
    print("-"*70)
    all_params = fitter.fit_all_components(
        segmented_ecg,
        use_global_optimization=True,
        optimization_method='differential_evolution',
        epsilon=0.01
    )
    
    # Print detailed results
    print("\n" + "="*70)
    print("FINAL GAUSSIAN PARAMETERS")
    print("="*70)
    for comp, params in all_params.items():
        print(f"\n{comp} Component:")
        print(f"  Gaussian 1: A1={params['A1']:7.4f}, t1={params['t1']:6.2f}, σ1={params['sigma1']:6.4f}")
        print(f"  Gaussian 2: A2={params['A2']:7.4f}, t2={params['t2']:6.2f}, σ2={params['sigma2']:6.4f}")
        print(f"  Baseline:   C={params['C']:7.4f}")
        print(f"  RMSE:       {params['rmse']:.8f}")
    
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    # Plot individual components
    print("\n1. Plotting individual components...")
    fitter.plot_individual_components(segmented_ecg, all_params)
    
    # Plot complete beat
    print("\n2. Plotting complete ECG beat...")
    overall_rmse = fitter.plot_complete_beat(segmented_ecg, all_params)
    
    # Plot summary table
    print("\n3. Creating parameters summary table...")
    fitter.plot_summary_table(all_params)
    
    print("\n" + "="*70)
    print(f"COMPLETE! Overall ECG Beat RMSE: {overall_rmse:.8f}")
    print("="*70)
    print("\nGenerated Files:")
    print("  • individual_components.png")
    print("  • complete_ecg_beat.png")
    print("  • parameters_summary.png")
    print("="*70)
    
    return all_params, overall_rmse


if __name__ == "__main__":
    params, rmse = example_usage()