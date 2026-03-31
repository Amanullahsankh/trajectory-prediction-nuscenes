# src/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class SocialLSTM(nn.Module):

    def __init__(self, obs_len=8, pred_len=12, embed_dim=64,
                 hidden_dim=128, num_modes=3,
                 num_neighbors=4, dropout=0.1):
        super(SocialLSTM, self).__init__()

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.hidden_dim = hidden_dim
        self.num_modes = num_modes
        self.num_neighbors = num_neighbors

        # Input: (x, y, vx, vy)
        self.input_embed = nn.Sequential(
            nn.Linear(4, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.encoder_lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )

        neighbor_input_dim = num_neighbors * hidden_dim
        self.social_pool = nn.Sequential(
            nn.Linear(neighbor_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.decoder_lstm = nn.LSTM(
            input_size=embed_dim + hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )

        # Output: only (x, y)
        self.output_heads = nn.ModuleList([
            nn.Linear(hidden_dim, 2) for _ in range(num_modes)
        ])

        self.mode_weights = nn.Linear(hidden_dim, num_modes)

    def encode(self, obs):
        embedded = self.input_embed(obs)
        enc_out, (h_n, c_n) = self.encoder_lstm(embedded)
        return enc_out, h_n, c_n

    def social_context(self, enc_out_neighbors):
        B = enc_out_neighbors.size(0)
        flat = enc_out_neighbors.reshape(B, -1)
        return self.social_pool(flat)

    def decode(self, h_n, c_n, social_ctx, last_obs):

        B = h_n.size(1)
        all_mode_preds = []

        for mode_idx in range(self.num_modes):

            dec_h, dec_c = h_n.clone(), c_n.clone()

            current_pos = last_obs[:, :2]
            current_vel = last_obs[:, 2:]

            preds = []

            for t in range(self.pred_len):

                model_input = torch.cat([current_pos, current_vel], dim=-1)
                pos_embed = self.input_embed(model_input)

                dec_input = torch.cat(
                    [pos_embed, social_ctx], dim=-1
                ).unsqueeze(1)

                dec_out, (dec_h, dec_c) = self.decoder_lstm(
                    dec_input, (dec_h, dec_c))

                delta = self.output_heads[mode_idx](
                    dec_out.squeeze(1))

                next_pos = current_pos + delta

                # Update velocity
                current_vel = next_pos - current_pos
                current_pos = next_pos

                preds.append(current_pos)

            preds = torch.stack(preds, dim=1)
            all_mode_preds.append(preds)

        predictions = torch.stack(all_mode_preds, dim=1)

        mode_scores = self.mode_weights(h_n[-1])

        return predictions, mode_scores

    def forward(self, obs, neighbors):

        B = obs.size(0)

        # Encode main agent
        enc_out, h_n, c_n = self.encode(obs)

        # Encode neighbors
        neighbor_hidden = []
        for n in range(self.num_neighbors):
            nb_obs = neighbors[:, n, :, :]

            # Pad neighbors to 4D (vx, vy = 0)
            zeros = torch.zeros_like(nb_obs)
            nb_input = torch.cat([nb_obs, zeros], dim=-1)

            _, nb_h, _ = self.encode(nb_input)
            neighbor_hidden.append(nb_h[-1])

        nb_hidden = torch.stack(neighbor_hidden, dim=1)

        social_ctx = self.social_context(nb_hidden)

        last_obs = obs[:, -1, :]

        predictions, mode_scores = self.decode(
            h_n, c_n, social_ctx, last_obs)

        mode_probs = F.softmax(mode_scores, dim=-1)

        return predictions, mode_probs


# ──────────────────────────────────────────────
# 🔥 IMPROVED LOSS FUNCTION (Soft Best-of-K)
# ──────────────────────────────────────────────

def improved_best_of_k_loss(predictions, mode_probs, ground_truth, temperature=0.5):
    """
    predictions  : (B, K, T, 2)
    mode_probs   : (B, K)
    ground_truth : (B, T, 2)
    """

    B, K, T, _ = predictions.shape

    # Expand GT
    gt = ground_truth.unsqueeze(1).expand_as(predictions)

    # Compute distance (ADE per mode)
    diff = predictions - gt
    dist = torch.norm(diff, dim=-1).mean(dim=-1)   # (B, K)

    # 🔥 Soft weighting
    weights = torch.softmax(-dist / temperature, dim=1)

    # Weighted regression
    regression_loss = (weights * dist).sum(dim=1).mean()

    # Classification loss
    best_mode = dist.argmin(dim=1)
    cls_loss = F.cross_entropy(mode_probs, best_mode)

    return regression_loss + 0.3 * cls_loss