import os
import json
from datetime import datetime
from collections import defaultdict, deque
import torch
import numpy as np
import matplotlib.pyplot as plt

# Class to track metrics with moving averages
class MetricTracker:
    """
    Track metrics with moving averages and historical data
    """
    def __init__(self, window_size=100):
        self.metrics = defaultdict(lambda: deque(maxlen=window_size))
        self.window_size = window_size
        self.all_metrics = defaultdict(list)  # Store all metrics for plotting

    def update(self, metric_dict):
        for key, value in metric_dict.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            self.metrics[key].append(value)
            self.all_metrics[key].append(value)

    def get_average(self, key):
        values = self.metrics[key]
        if not values:
            return 0.0
        return sum(values) / len(values)

    def get_all_metrics(self):
        return {k: self.get_average(k) for k in self.metrics.keys()}
    
def restore_metrics_tracker(metrics_history, window_size=100):
    """
    Restore metrics tracker from saved metrics history
    
    Args:
        metrics_history: Dictionary with metrics history
        window_size: Window size for moving averages
    
    Returns:
        Restored MetricTracker object
    """
    tracker = MetricTracker(window_size=window_size)
    
    # Restore all metrics history
    for key, values in metrics_history.items():
        tracker.all_metrics[key] = values
        
        # Restore recent metrics window (last window_size items)
        for value in values[-window_size:]:
            tracker.metrics[key].append(value)
    
    return tracker

# Visualization function for metrics
def plot_metrics(tracker, step, save_dir="./visualizations", experiment_name=None):
    """
    Plot training metrics and save visualization to the specified directory.
    Also saves metrics data in JSON format.
    
    Args:
        tracker: MetricTracker object with metrics history
        step: Current training step
        save_dir: Directory to save visualizations
        experiment_name: Optional experiment name for creating subdirectories
        
    Returns:
        Path to the saved visualization file
    """
    # Create directory structure if it doesn't exist
    if experiment_name:
        # Use experiment-specific subdirectory
        vis_path = os.path.join(save_dir, experiment_name)
    else:
        vis_path = save_dir
    
    os.makedirs(vis_path, exist_ok=True)
    
    # Save metrics data to JSON file
    metrics_data = {
        'step': step,
        'timestamp': datetime.now().strftime("%Y%m%d-%H%M%S"),
        'metrics': {}
    }
    
    # Add all metrics to the JSON data
    for metric_name, metric_values in tracker.all_metrics.items():
        metrics_data['metrics'][metric_name] = {
            'values': metric_values,
            'latest': metric_values[-1] if metric_values else None,
            'average': sum(metric_values[-min(30, len(metric_values)):]) / min(30, len(metric_values)) 
                      if metric_values else None
        }
    
    # Save JSON data
    json_file_path = os.path.join(vis_path, f"metrics_data_step_{step}.json")
    with open(json_file_path, 'w', encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)
    
    # Create a 2x4 grid of plots (changing from 4x2 to 2x4 layout)
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f'Training Metrics at Step {step}', fontsize=16)
    
    # Plot answer accuracy metrics
    if 'answer_accuracy' in tracker.all_metrics:
        axs[0, 0].plot(tracker.all_metrics['answer_accuracy'])
        axs[0, 0].set_title('Answer Accuracy')
        axs[0, 0].set_xlabel('Steps')
        axs[0, 0].set_ylabel('Accuracy')
        
        # Plot moving average
        window_size = min(30, len(tracker.all_metrics['answer_accuracy']))
        if window_size > 0:
            moving_avg = np.convolve(tracker.all_metrics['answer_accuracy'], 
                                    np.ones(window_size)/window_size, 
                                    mode='valid')
            axs[0, 0].plot(range(window_size-1, len(tracker.all_metrics['answer_accuracy'])), 
                           moving_avg, 'r-', label=f'MA-{window_size}')
        axs[0, 0].legend()
    
    # Plot format accuracy
    if 'format_accuracy' in tracker.all_metrics:
        axs[0, 1].plot(tracker.all_metrics['format_accuracy'])
        axs[0, 1].set_title('Format Accuracy')
        axs[0, 1].set_xlabel('Steps')
        axs[0, 1].set_ylabel('Accuracy')
        
        # Plot moving average
        window_size = min(30, len(tracker.all_metrics['format_accuracy']))
        if window_size > 0:
            moving_avg = np.convolve(tracker.all_metrics['format_accuracy'], 
                                    np.ones(window_size)/window_size, 
                                    mode='valid')
            axs[0, 1].plot(range(window_size-1, len(tracker.all_metrics['format_accuracy'])), 
                           moving_avg, 'r-', label=f'MA-{window_size}')
        axs[0, 1].legend()
    
    # Plot combined score
    if 'combined_score' in tracker.all_metrics:
        axs[0, 2].plot(tracker.all_metrics['combined_score'])
        axs[0, 2].set_title('Combined Score')
        axs[0, 2].set_xlabel('Steps')
        axs[0, 2].set_ylabel('Score')
        
        # Plot moving average
        window_size = min(30, len(tracker.all_metrics['combined_score']))
        if window_size > 0:
            moving_avg = np.convolve(tracker.all_metrics['combined_score'], 
                                    np.ones(window_size)/window_size, 
                                    mode='valid')
            axs[0, 2].plot(range(window_size-1, len(tracker.all_metrics['combined_score'])), 
                           moving_avg, 'r-', label=f'MA-{window_size}')
        axs[0, 2].legend()
    
    # Plot DPO loss
    if 'dpo_loss' in tracker.all_metrics:
        axs[0, 3].plot(tracker.all_metrics['dpo_loss'])
        axs[0, 3].set_title('DPO Loss')
        axs[0, 3].set_xlabel('Steps')
        axs[0, 3].set_ylabel('Loss')
        
        # Add moving average for loss
        window_size = min(30, len(tracker.all_metrics['dpo_loss']))
        if window_size > 0:
            moving_avg = np.convolve(tracker.all_metrics['dpo_loss'], 
                                    np.ones(window_size)/window_size, 
                                    mode='valid')
            axs[0, 3].plot(range(window_size-1, len(tracker.all_metrics['dpo_loss'])), 
                          moving_avg, 'r-', label=f'MA-{window_size}')
        axs[0, 3].legend()
    
    # Plot SFT loss if available
    if 'sft_loss' in tracker.all_metrics:
        axs[1, 0].plot(tracker.all_metrics['sft_loss'])
        axs[1, 0].set_title('SFT Loss')
        axs[1, 0].set_xlabel('Steps')
        axs[1, 0].set_ylabel('Loss')
        
        # Add moving average for SFT loss
        window_size = min(30, len(tracker.all_metrics['sft_loss']))
        if window_size > 0:
            moving_avg = np.convolve(tracker.all_metrics['sft_loss'], 
                                    np.ones(window_size)/window_size, 
                                    mode='valid')
            axs[1, 0].plot(range(window_size-1, len(tracker.all_metrics['sft_loss'])), 
                          moving_avg, 'r-', label=f'MA-{window_size}')
        axs[1, 0].legend()
    else:
        # If no SFT loss, plot bucket list size
        if 'bucket_list_size' in tracker.all_metrics:
            axs[1, 0].plot(tracker.all_metrics['bucket_list_size'])
            axs[1, 0].set_title('Bucket List Size')
            axs[1, 0].set_xlabel('Steps')
            axs[1, 0].set_ylabel('Size')
    
    # Plot average verification attempts per problem
    if 'verification_attempts' in tracker.all_metrics:
        axs[1, 1].plot(tracker.all_metrics['verification_attempts'])
        axs[1, 1].set_title('Avg Verification Attempts')
        axs[1, 1].set_xlabel('Steps')
        axs[1, 1].set_ylabel('Attempts')
        
        # Add moving average for verification attempts
        window_size = min(30, len(tracker.all_metrics['verification_attempts']))
        if window_size > 0:
            moving_avg = np.convolve(tracker.all_metrics['verification_attempts'], 
                                    np.ones(window_size)/window_size, 
                                    mode='valid')
            axs[1, 1].plot(range(window_size-1, len(tracker.all_metrics['verification_attempts'])), 
                          moving_avg, 'r-', label=f'MA-{window_size}')
        axs[1, 1].legend()
    
    # Plot First Time Trial Accuracy
    if 'first_time_accuracy' in tracker.all_metrics:
        axs[1, 2].plot(tracker.all_metrics['first_time_accuracy'])
        axs[1, 2].set_title('First Time Trial Accuracy')
        axs[1, 2].set_xlabel('Steps')
        axs[1, 2].set_ylabel('Accuracy')
        
        # Add moving average
        window_size = min(30, len(tracker.all_metrics['first_time_accuracy']))
        if window_size > 0:
            moving_avg = np.convolve(tracker.all_metrics['first_time_accuracy'], 
                                    np.ones(window_size)/window_size, 
                                    mode='valid')
            axs[1, 2].plot(range(window_size-1, len(tracker.all_metrics['first_time_accuracy'])), 
                          moving_avg, 'r-', label=f'MA-{window_size}')
        axs[1, 2].legend()
        
    # Plot Bucket Size Progress
    if 'bucket_size_progress' in tracker.all_metrics:
        axs[1, 3].plot(tracker.all_metrics['bucket_size_progress'])
        axs[1, 3].set_title('Bucket Size Progress')
        axs[1, 3].set_xlabel('Steps')
        axs[1, 3].set_ylabel('Ratio')
        
        # Add moving average
        window_size = min(30, len(tracker.all_metrics['bucket_size_progress']))
        if window_size > 0:
            moving_avg = np.convolve(tracker.all_metrics['bucket_size_progress'], 
                                    np.ones(window_size)/window_size, 
                                    mode='valid')
            axs[1, 3].plot(range(window_size-1, len(tracker.all_metrics['bucket_size_progress'])), 
                          moving_avg, 'r-', label=f'MA-{window_size}')
        axs[1, 3].legend()
        
    plt.tight_layout()
    
    # Save with standardized naming including step number
    file_path = os.path.join(vis_path, f"metrics_step_{step}.png")
    plt.savefig(file_path)
    plt.close()
    
    return file_path  # Return the path to the saved file