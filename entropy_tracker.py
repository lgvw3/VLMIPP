from matplotlib import pyplot as plt
import numpy as np


def compute_spatial_entropy_distribution(entropy_map, obstacle_mask):
    """Analyze spatial distribution of entropy"""
    valid_entropy = entropy_map[~obstacle_mask]
    return {
        'mean_entropy': np.mean(valid_entropy),
        'std_entropy': np.std(valid_entropy),
        'max_entropy': np.max(valid_entropy),
        'min_entropy': np.min(valid_entropy),
        'entropy_percentiles': {
            '25%': np.percentile(valid_entropy, 25),
            '50%': np.percentile(valid_entropy, 50),
            '75%': np.percentile(valid_entropy, 75),
            '90%': np.percentile(valid_entropy, 90)
        }
    }

def compute_entropy_convergence_metrics(entropy_history, threshold=0.01):
    """Analyze convergence behavior"""
    if len(entropy_history) < 2:
        return {}
    
    # Find when entropy reduction becomes small
    convergence_step = None
    for i in range(1, len(entropy_history)):
        if abs(entropy_history[i-1] - entropy_history[i]) < threshold:
            convergence_step = i
            break
    
    # Calculate convergence rate
    if convergence_step and convergence_step > 1:
        initial_rate = entropy_history[0] - entropy_history[1]
        final_rate = entropy_history[convergence_step-1] - entropy_history[convergence_step]
        rate_decay = initial_rate / final_rate if final_rate > 0 else float('inf')
    else:
        rate_decay = None
    
    return {
        'convergence_step': convergence_step,
        'convergence_threshold': threshold,
        'rate_decay_factor': rate_decay,
        'final_entropy_level': entropy_history[-1] if entropy_history else None
    }


class EntropyTracker:
    def __init__(self):
        self.entropy_history = []
        self.entropy_reduction_history = []
        self.entropy_rate_history = []
        self.step_times = []
        self.total_observations_history = []
        
    def update(self, total_entropy, time_step, num_observations):
        """Update entropy tracking with new measurements"""
        self.entropy_history.append(total_entropy)
        self.step_times.append(time_step)
        self.total_observations_history.append(num_observations)
        
        if len(self.entropy_history) > 1:
            # Calculate entropy reduction (previous - current)
            reduction = self.entropy_history[-2] - self.entropy_history[-1]
            self.entropy_reduction_history.append(reduction)
            
            # Calculate entropy rate (over last 5 steps if available)
            if len(self.entropy_history) >= 5:
                recent_entropy = self.entropy_history[-5:]
                rate = (recent_entropy[0] - recent_entropy[-1]) / 4
                self.entropy_rate_history.append(rate)
            else:
                self.entropy_rate_history.append(0.0)
        else:
            self.entropy_reduction_history.append(0.0)
            self.entropy_rate_history.append(0.0)
    
    def get_summary_stats(self):
        """Get comprehensive summary statistics"""
        if len(self.entropy_history) < 2:
            return {}
        
        total_reduction = self.entropy_history[0] - self.entropy_history[-1]
        avg_reduction_per_step = total_reduction / (len(self.entropy_history) - 1)
        
        # Calculate efficiency metrics
        total_observations = self.total_observations_history[-1] if self.total_observations_history else 0
        efficiency_per_observation = total_reduction / total_observations if total_observations > 0 else 0
        
        # Find convergence point (when entropy reduction becomes small)
        convergence_step = None
        for i, reduction in enumerate(self.entropy_reduction_history):
            if abs(reduction) < 0.01:  # Threshold for convergence
                convergence_step = i
                break
        
        return {
            'total_entropy_reduction': total_reduction,
            'avg_reduction_per_step': avg_reduction_per_step,
            'efficiency_per_observation': efficiency_per_observation,
            'final_entropy': self.entropy_history[-1],
            'initial_entropy': self.entropy_history[0],
            'convergence_step': convergence_step,
            'total_steps': len(self.entropy_history),
            'total_observations': total_observations
        }
    
    def plot_entropy_trajectory(self, save_path=None):
        """Plot comprehensive entropy analysis"""
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Total entropy over time
        axs[0, 0].plot(self.step_times, self.entropy_history, 'b-', linewidth=2, marker='o')
        axs[0, 0].set_title('Total Entropy Over Time')
        axs[0, 0].set_xlabel('Time Step')
        axs[0, 0].set_ylabel('Total Entropy')
        axs[0, 0].grid(True)
        
        # Plot 2: Entropy reduction per step
        if len(self.entropy_reduction_history) > 1:
            axs[0, 1].plot(self.step_times, self.entropy_reduction_history, 'r-', linewidth=2, marker='s')
            axs[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        axs[0, 1].set_title('Entropy Reduction Per Step')
        axs[0, 1].set_xlabel('Time Step')
        axs[0, 1].set_ylabel('Entropy Reduction')
        axs[0, 1].grid(True)
        
        # Plot 3: Entropy rate (rolling average)
        if len(self.entropy_rate_history) > 1:
            axs[1, 0].plot(self.step_times, self.entropy_rate_history, 'g-', linewidth=2, marker='^')
        axs[1, 0].set_title('Entropy Reduction Rate (5-step average)')
        axs[1, 0].set_xlabel('Time Step')
        axs[1, 0].set_ylabel('Entropy Reduction Rate')
        axs[1, 0].grid(True)
        
        # Plot 4: Observations vs Entropy
        axs[1, 1].plot(self.total_observations_history, self.entropy_history, 'purple', linewidth=2, marker='d')
        axs[1, 1].set_title('Entropy vs Total Observations')
        axs[1, 1].set_xlabel('Total Observations')
        axs[1, 1].set_ylabel('Total Entropy')
        axs[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()