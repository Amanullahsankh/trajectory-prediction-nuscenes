# src/evaluate.py

import torch
from tqdm import tqdm

from model import SocialLSTM
from utils import compute_ade, compute_fde, get_device
from src.dataset import get_dataloaders


@torch.no_grad()
def evaluate(model_path, config=None):

    device = get_device()
    print(f"Using device: {device}")

    checkpoint = torch.load(model_path, map_location=device)

    if config is None:
        config = checkpoint['config']

    # Load data
    _, _, test_loader = get_dataloaders(config)

    # Load model
    model = SocialLSTM(
        obs_len=config['obs_len'],
        pred_len=config['pred_len'],
        embed_dim=config['embed_dim'],
        hidden_dim=config['hidden_dim'],
        num_modes=config['num_modes'],
        num_neighbors=config['num_neighbors'],
        dropout=config['dropout']
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Loaded model from epoch {checkpoint['epoch']}")

    total_ade, total_fde = 0.0, 0.0

    for batch in tqdm(test_loader, desc="Evaluating"):
        obs, future, neighbors = [x.to(device) for x in batch]

        predictions, mode_probs = model(obs, neighbors)

        # 🔥 FIX: use only (x, y)
        gt = future[:, :, :2]

        # Best mode
        best_mode = mode_probs.argmax(dim=1)

        best_pred = predictions[
            torch.arange(obs.size(0)), best_mode
        ]

        total_ade += compute_ade(best_pred, gt)
        total_fde += compute_fde(best_pred, gt)

    n = len(test_loader)

    avg_ade = total_ade / n
    avg_fde = total_fde / n

    print("\n" + "="*50)
    print(f"Test ADE: {avg_ade:.4f}")
    print(f"Test FDE: {avg_fde:.4f}")
    print("="*50)

    return avg_ade, avg_fde