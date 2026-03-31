# src/utils.py
# Utility functions used across the project

import numpy as np
import torch
import matplotlib.pyplot as plt


def compute_ade(predicted, ground_truth):
    """
    Average Displacement Error (ADE)
    predicted:     shape (batch, future_steps, 2)
    ground_truth:  shape (batch, future_steps, 2)
    Returns: scalar ADE value
    """
    # Euclidean distance at each timestep, then mean over all steps and agents
    diff = predicted - ground_truth                        # (batch, steps, 2)
    dist = torch.sqrt((diff ** 2).sum(dim=-1))             # (batch, steps)
    ade = dist.mean()
    return ade.item()


def compute_fde(predicted, ground_truth):
    """
    Final Displacement Error (FDE)
    Only looks at the LAST predicted point vs the last ground truth point.
    predicted:     shape (batch, future_steps, 2)
    ground_truth:  shape (batch, future_steps, 2)
    Returns: scalar FDE value
    """
    pred_final = predicted[:, -1, :]       # (batch, 2)
    gt_final = ground_truth[:, -1, :]      # (batch, 2)
    diff = pred_final - gt_final
    dist = torch.sqrt((diff ** 2).sum(dim=-1))
    fde = dist.mean()
    return fde.item()


def set_seed(seed=42):
    """Fix random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def get_device():
    """Auto-detect GPU or fall back to CPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def plot_trajectories(history, ground_truth, predictions, sample_idx=0):
    """
    Visualize one sample: past path, true future, predicted futures.
    history:      (obs_len, 2)
    ground_truth: (pred_len, 2)
    predictions:  (num_modes, pred_len, 2)
    """
    plt.figure(figsize=(8, 8))

    # History (blue)
    plt.plot(history[:, 0], history[:, 1],
             'bo-', label='History (2s)', linewidth=2, markersize=6)

    # Ground truth future (green)
    plt.plot(ground_truth[:, 0], ground_truth[:, 1],
             'go-', label='Ground Truth (3s)', linewidth=2, markersize=6)

    # Predicted futures (red, one line per mode)
    colors = ['r', 'orange', 'purple']
    for i, pred in enumerate(predictions):
        plt.plot(pred[:, 0], pred[:, 1],
                 color=colors[i % len(colors)],
                 linestyle='--', marker='x',
                 label=f'Prediction Mode {i+1}',
                 linewidth=1.5)

    plt.legend()
    plt.title(f'Trajectory Prediction — Sample {sample_idx}')
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(f'prediction_sample_{sample_idx}.png', dpi=150)
    plt.show()