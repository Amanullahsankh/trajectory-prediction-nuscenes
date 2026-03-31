# src/dataset.py

import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Try importing nuScenes
try:
    from nuscenes.nuscenes import NuScenes
    NUSCENES_AVAILABLE = True
except:
    NUSCENES_AVAILABLE = False


# ──────────────────────────────────────────────
# 🔧 HELPER: VELOCITY FUNCTION
# ──────────────────────────────────────────────

def compute_velocity(traj):
    """
    traj: (T, 2)
    returns: (T, 2) velocity
    """
    vel = np.zeros_like(traj)
    vel[1:] = traj[1:] - traj[:-1]
    vel[0] = vel[1]  # avoid zero at start
    return vel


# ──────────────────────────────────────────────
# REAL DATASET (nuScenes)
# ──────────────────────────────────────────────

class NuScenesDataset(Dataset):
    def __init__(self, data_path, obs_len=8, pred_len=12, num_neighbors=4):
        self.data_path = data_path
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.num_neighbors = num_neighbors

        if not NUSCENES_AVAILABLE:
            raise ImportError("nuscenes-devkit not installed")

        print("Loading nuScenes dataset...")

        self.nusc = NuScenes(
            version='v1.0-mini',
            dataroot=data_path,
            verbose=True
        )

        self.samples = self.nusc.sample

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        try:
            anns = sample['anns']
            traj = []

            # Collect pedestrian positions
            for ann_token in anns:
                ann = self.nusc.get('sample_annotation', ann_token)
                if "pedestrian" in ann['category_name']:
                    traj.append(ann['translation'][:2])

            if len(traj) < (self.obs_len + self.pred_len):
                raise ValueError("Not enough trajectory points")

            traj = np.array(traj[:self.obs_len + self.pred_len])

            # Split
            obs_np = traj[:self.obs_len]
            future_np = traj[self.obs_len:]

        except:
            # Safe fallback
            obs_np = np.random.randn(self.obs_len, 2)
            future_np = np.random.randn(self.pred_len, 2)

        # 🔥 Compute velocity
        obs_vel = compute_velocity(obs_np)
        future_vel = compute_velocity(future_np)

        # 🔥 Concatenate position + velocity
        obs = torch.FloatTensor(np.concatenate([obs_np, obs_vel], axis=1))       # (T, 4)
        future = torch.FloatTensor(np.concatenate([future_np, future_vel], axis=1))

        # Dummy neighbors (still position only for now)
        neighbors = torch.randn(self.num_neighbors, self.obs_len, 2)

        # 🔥 Normalize ONLY position (not velocity)
        origin = obs[-1:, :2]

        obs[:, :2] = obs[:, :2] - origin
        future[:, :2] = future[:, :2] - origin
        neighbors = neighbors - origin.unsqueeze(0)

        return obs, future, neighbors


# ──────────────────────────────────────────────
# SYNTHETIC DATASET (fallback)
# ──────────────────────────────────────────────

class SyntheticTrajectoryDataset(Dataset):
    def __init__(self, num_agents=1000, obs_len=8, pred_len=12,
                 num_neighbors=4, normalize=True):
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.num_neighbors = num_neighbors
        self.normalize = normalize
        self.total_len = obs_len + pred_len

        self.data = self._generate(num_agents)

    def _generate(self, num_agents):
        data = []
        for _ in range(num_agents):
            start = np.random.uniform(-25, 25, size=(1, 2))
            velocity = np.random.uniform(-1.5, 1.5, size=(1, 2))

            trajectory = [start[0]]
            for _ in range(1, self.total_len):
                noise = np.random.normal(0, 0.05, size=2)
                trajectory.append(trajectory[-1] + velocity[0] + noise)

            trajectory = np.array(trajectory)

            neighbors = []
            for _ in range(self.num_neighbors):
                offset = np.random.uniform(-5, 5, size=(1, 2))
                nb_vel = np.random.uniform(-1.5, 1.5, size=(1, 2))

                nb = [trajectory[0] + offset[0]]
                for _ in range(1, self.obs_len):
                    nb_noise = np.random.normal(0, 0.05, size=2)
                    nb.append(nb[-1] + nb_vel[0] + nb_noise)

                neighbors.append(np.array(nb))

            neighbors = np.stack(neighbors, axis=0)

            data.append({
                'obs': trajectory[:self.obs_len],
                'future': trajectory[self.obs_len:],
                'neighbors': neighbors,
            })

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        obs_np = sample['obs']
        future_np = sample['future']
        neighbors = torch.FloatTensor(sample['neighbors'])

        # 🔥 Compute velocity
        obs_vel = compute_velocity(obs_np)
        future_vel = compute_velocity(future_np)

        # 🔥 Concatenate
        obs = torch.FloatTensor(np.concatenate([obs_np, obs_vel], axis=1))
        future = torch.FloatTensor(np.concatenate([future_np, future_vel], axis=1))

        if self.normalize:
            origin = obs[-1:, :2]

            obs[:, :2] = obs[:, :2] - origin
            future[:, :2] = future[:, :2] - origin
            neighbors = neighbors - origin.unsqueeze(0)

        return obs, future, neighbors


# ──────────────────────────────────────────────
# DATALOADER
# ──────────────────────────────────────────────

def get_dataloaders(config):
    data_path = config.get('data_path', None)

    use_nuscenes = (
        data_path is not None and
        os.path.exists(data_path) and
        NUSCENES_AVAILABLE
    )

    if use_nuscenes:
        print("✅ Using nuScenes dataset")

        train_set = NuScenesDataset(
            data_path,
            obs_len=config['obs_len'],
            pred_len=config['pred_len'],
            num_neighbors=config['num_neighbors']
        )

        val_set = train_set
        test_set = train_set

    else:
        print("⚠️ Using synthetic dataset (fallback)")
        print("👉 Fix this by setting correct data_path in config")

        train_set = SyntheticTrajectoryDataset(
            num_agents=8000,
            obs_len=config['obs_len'],
            pred_len=config['pred_len'],
            num_neighbors=config['num_neighbors']
        )

        val_set = SyntheticTrajectoryDataset(
            num_agents=1000,
            obs_len=config['obs_len'],
            pred_len=config['pred_len'],
            num_neighbors=config['num_neighbors']
        )

        test_set = SyntheticTrajectoryDataset(
            num_agents=1000,
            obs_len=config['obs_len'],
            pred_len=config['pred_len'],
            num_neighbors=config['num_neighbors']
        )

    train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=config['batch_size'], shuffle=False)
    test_loader  = DataLoader(test_set,  batch_size=config['batch_size'], shuffle=False)

    print(f"Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}")

    return train_loader, val_loader, test_loader