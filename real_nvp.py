import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import time
import random
import argparse
from tqdm import trange

from metrics import Metrics
from datasets import TwoDDatasets

# ============================================================
# Seed fixing
# ============================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed')
args = parser.parse_args()


# ============================================================
# Settings
# ============================================================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
output_dir = 'outputs/realnvp_outputs'
os.makedirs(output_dir, exist_ok=True)

datasets_names = ['two_moons', 'gaussian_mixture', 'concentric_circles']

configs = [
    dict(n_layers=4, hidden_dim=64, use_batchnorm=False, lr=1e-3, n_epochs=200),
    dict(n_layers=6, hidden_dim=64, use_batchnorm=False, lr=1e-3, n_epochs=200),
    dict(n_layers=6, hidden_dim=64, use_batchnorm=False, lr=1e-3, n_epochs=200),
    dict(n_layers=6, hidden_dim=128, use_batchnorm=False, lr=1e-3, n_epochs=200),
    dict(n_layers=6, hidden_dim=128, use_batchnorm=False, lr=1e-3, n_epochs=200),
    dict(n_layers=6, hidden_dim=128, use_batchnorm=True, lr=5e-4, n_epochs=200),
]

batch_size = 256
latent_dim = 2

# ============================================================
# RealNVP building blocks
# ============================================================
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, use_batchnorm=False):
        super().__init__()
        layers = [
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
        ]
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers += [
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class CouplingLayer(nn.Module):
    def __init__(self, dim, mask, hidden_dim, use_batchnorm):
        super().__init__()
        self.mask = mask
        self.scale_net = MLP(dim, dim, hidden_dim, use_batchnorm)
        self.shift_net = MLP(dim, dim, hidden_dim, use_batchnorm)

    def forward(self, x):
        x_masked = x * self.mask
        s = self.scale_net(x_masked) * (1 - self.mask)
        t = self.shift_net(x_masked) * (1 - self.mask)
        y = x_masked + (1 - self.mask) * (x * torch.exp(s) + t)
        log_det = s.sum(dim=1)
        return y, log_det

    def inverse(self, y):
        y_masked = y * self.mask
        s = self.scale_net(y_masked) * (1 - self.mask)
        t = self.shift_net(y_masked) * (1 - self.mask)
        x = y_masked + (1 - self.mask) * (y - t) * torch.exp(-s)
        log_det = -s.sum(dim=1)
        return x, log_det


class RealNVP(nn.Module):
    def __init__(self, dim, n_layers, hidden_dim, use_batchnorm):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            mask = torch.tensor([i % 2, (i + 1) % 2], dtype=torch.float32)
            self.layers.append(
                CouplingLayer(dim, mask.to(device), hidden_dim, use_batchnorm)
            )

    def forward(self, x):
        log_det = 0
        for layer in self.layers:
            x, ld = layer(x)
            log_det += ld
        return x, log_det

    def inverse(self, z):
        log_det = 0
        for layer in reversed(self.layers):
            z, ld = layer.inverse(z)
            log_det += ld
        return z, log_det


# ============================================================
# Mode coverage
# ============================================================
def compute_mode_coverage(dataset_name, real, gen, r=0.3):
    if dataset_name == 'gaussian_mixture':
        angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
        centers = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    elif dataset_name == 'two_moons':
        centers = np.array([[0, 0.5], [0, -0.5]])
    elif dataset_name == 'concentric_circles':
        centers = np.array([[0, 0]])
    else:
        return 0.0

    covered = 0
    for c in centers:
        if np.any(np.linalg.norm(gen - c, axis=1) < r):
            covered += 1
    return covered / len(centers)


# ============================================================
# Training
# ============================================================
datasets = TwoDDatasets(n_samples=10000).create_all_datasets()
metrics = Metrics(device=device)

for dataset_id, dataset_name in enumerate(datasets_names):
    for cfg_id, cfg in enumerate(configs):

        # уникальный seed для каждой комбинации
        set_seed(args.seed)

        print(f"\n=== RealNVP | {dataset_name} | cfg {cfg_id+1} | seed {args.seed} ===")

        data = datasets[dataset_name]['data'].to(device)

        model = RealNVP(
            dim=2,
            n_layers=cfg['n_layers'],
            hidden_dim=cfg['hidden_dim'],
            use_batchnorm=cfg['use_batchnorm']
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=cfg['lr'])
        metrics_history = []

        start_time = time.time()

        for epoch in trange(1, cfg['n_epochs'] + 1):
            perm = torch.randperm(data.size(0))
            for i in range(0, data.size(0), batch_size):
                batch = data[perm[i:i+batch_size]]
                if batch.size(0) < batch_size:
                    continue

                z, log_det = model(batch)
                log_prob = -0.5 * (z ** 2).sum(dim=1)
                loss = -(log_prob + log_det).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if epoch % 50 == 0 or epoch == 1:
                with torch.no_grad():
                    z = torch.randn(1000, 2, device=device)
                    samples, _ = model.inverse(z)
                    samples = samples.cpu().numpy()

                real_np = data[:1000].cpu().numpy()

                metric_vals = metrics.compute_all_metrics(
                    real_np, samples, true_distribution=dataset_name
                )
                metric_vals['epoch'] = epoch
                metric_vals['mode_coverage'] = compute_mode_coverage(
                    dataset_name, real_np, samples
                )

                metrics_history.append(metric_vals)

                plt.figure(figsize=(5,5))
                plt.scatter(real_np[:,0], real_np[:,1], alpha=0.3, label='Real')
                plt.scatter(samples[:,0], samples[:,1], alpha=0.5, label='Generated')
                plt.legend()
                plt.title(f"{dataset_name} | cfg {cfg_id+1} | epoch {epoch}")
                plt.savefig(f"{output_dir}/{dataset_name}_cfg{cfg_id+1}_epoch{epoch}.png")
                plt.close()

        total_time = time.time() - start_time

        df = pd.DataFrame(metrics_history)
        df['dataset'] = dataset_name
        df['config_id'] = cfg_id + 1
        df['seed'] = args.seed
        df['n_layers'] = cfg['n_layers']
        df['hidden_dim'] = cfg['hidden_dim']
        df['use_batchnorm'] = cfg['use_batchnorm']
        df['total_training_time'] = total_time

        df.to_csv(
            f"{output_dir}/metrics_{dataset_name}_cfg{cfg_id+1}.csv",
            index=False
        )

        print(f"Finished in {total_time:.2f}s")

print("\nAll RealNVP experiments finished")
