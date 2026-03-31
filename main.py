# main.py
# Run this file to train and evaluate the model!
# Usage: python main.py

import sys
import os
sys.path.insert(0, 'src')

from utils import set_seed

# ── Configuration ─────────────────────────────────────────────────────────────
CONFIG = {
    # Data
    'data_path'         : 'data/nuscenes',   # ✅ ADD THIS (VERY IMPORTANT)
    'obs_len'           : 8,
    'pred_len'          : 12,
    'num_neighbors'     : 4,

    # Model architecture
    'embed_dim'         : 64,
    'hidden_dim'        : 128,
    'num_modes'         : 3,
    'dropout'           : 0.1,

    # Training
    'batch_size'        : 64,
    'lr'                : 1e-3,
    'epochs'            : 50,
    'early_stop_patience': 10,
}

if __name__ == '__main__':
    set_seed(42)

    # ── CHECK DATA PATH ──────────────────────────────────────────────────
    if not os.path.exists(CONFIG['data_path']):
        raise FileNotFoundError(f"Dataset not found at {CONFIG['data_path']}")

    print(f"Dataset found at: {CONFIG['data_path']}")

    # ── TRAIN ────────────────────────────────────────────────────────────
    print("\nStep 1: Training the model...")
    from train import train
    model = train(CONFIG)

    # ── EVALUATE ─────────────────────────────────────────────────────────
    print("\nStep 2: Evaluating on test set...")
    from evaluate import evaluate

    model_path = 'checkpoints/best_model.pth'

    if not os.path.exists(model_path):
        print("⚠️ Model checkpoint not found, skipping evaluation.")
    else:
        evaluate(model_path)