# src/train.py

import torch
import torch.optim as optim
from tqdm import tqdm
import os

from model import SocialLSTM, improved_best_of_k_loss   # ✅ UPDATED
from utils import compute_ade, compute_fde, get_device
from src.dataset import get_dataloaders


# ──────────────────────────────────────────────
# TRAIN ONE EPOCH
# ──────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader, desc="Training", leave=False):
        obs, future, neighbors = [x.to(device) for x in batch]

        optimizer.zero_grad()

        predictions, mode_probs = model(obs, neighbors)

        # ✅ Use only (x, y)
        gt = future[:, :, :2]

        # 🔥 UPDATED LOSS
        loss = improved_best_of_k_loss(predictions, mode_probs, gt)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# ──────────────────────────────────────────────
# VALIDATION
# ──────────────────────────────────────────────
@torch.no_grad()
def validate(model, loader, device):
    model.eval()

    total_loss, total_ade, total_fde = 0.0, 0.0, 0.0

    for batch in tqdm(loader, desc="Validating", leave=False):
        obs, future, neighbors = [x.to(device) for x in batch]

        predictions, mode_probs = model(obs, neighbors)

        # ✅ Use only (x, y)
        gt = future[:, :, :2]

        # 🔥 UPDATED LOSS
        loss = improved_best_of_k_loss(predictions, mode_probs, gt)

        # Choose best mode
        best_mode = mode_probs.argmax(dim=1)

        best_pred = predictions[
            torch.arange(obs.size(0)), best_mode
        ]

        total_loss += loss.item()
        total_ade += compute_ade(best_pred, gt)
        total_fde += compute_fde(best_pred, gt)

    n = len(loader)
    return total_loss / n, total_ade / n, total_fde / n


# ──────────────────────────────────────────────
# MAIN TRAIN FUNCTION
# ──────────────────────────────────────────────
def train(config):

    device = get_device()

    # ── Data ────────────────────────────────────────────
    train_loader, val_loader, _ = get_dataloaders(config)

    # ── Model ───────────────────────────────────────────
    model = SocialLSTM(
        obs_len=config['obs_len'],
        pred_len=config['pred_len'],
        embed_dim=config['embed_dim'],
        hidden_dim=config['hidden_dim'],
        num_modes=config['num_modes'],
        num_neighbors=config['num_neighbors'],
        dropout=config['dropout']
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # ── Optimizer & Scheduler ───────────────────────────
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    patience=5,
    factor=0.5
)

    # ── Training Loop ───────────────────────────────────
    best_val_ade = float('inf')
    patience_counter = 0
    os.makedirs('checkpoints', exist_ok=True)

    print("\n" + "="*50)
    print("Starting Training")
    print("="*50)

    for epoch in range(1, config['epochs'] + 1):

        train_loss = train_one_epoch(model, train_loader, optimizer, device)

        val_loss, val_ade, val_fde = validate(model, val_loader, device)
        scheduler.step(val_loss)

        print(f"Epoch {epoch:3d}/{config['epochs']} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"ADE: {val_ade:.4f} | "
              f"FDE: {val_fde:.4f}")

        # ✅ Save best model
        if val_ade < best_val_ade:
            best_val_ade = val_ade
            patience_counter = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_ade': val_ade,
                'val_fde': val_fde,
                'config': config,
            }, 'checkpoints/best_model.pth')

            print(f"  ✓ New best model saved (ADE: {val_ade:.4f})")

        else:
            patience_counter += 1

            if patience_counter >= config['early_stop_patience']:
                print(f"\nEarly stopping triggered at epoch {epoch}")
                break

    print(f"\nTraining complete. Best Val ADE: {best_val_ade:.4f}")

    return model