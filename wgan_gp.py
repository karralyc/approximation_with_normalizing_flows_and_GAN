import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import time
import argparse
from tqdm import trange

from metrics import Metrics
from datasets import TwoDDatasets

# ============================================================
# Settings
# ============================================================

device = 'cuda' if torch.cuda.is_available() else 'cpu'
output_dir = 'outputs/wgan_outputs'
os.makedirs(output_dir, exist_ok=True)
configs = [
    {'latent_dim': 16, 'hidden_dim': 64, 'lr': 1e-4, 'batch_size': 128, 'n_epochs': 200},
    {'latent_dim': 16, 'hidden_dim': 128, 'lr': 1e-4, 'batch_size': 128, 'n_epochs': 200},
    {'latent_dim': 32, 'hidden_dim': 64, 'lr': 5e-4, 'batch_size': 128, 'n_epochs': 200},
    {'latent_dim': 32, 'hidden_dim': 128, 'lr': 5e-4, 'batch_size': 128, 'n_epochs': 200},
]

datasets_names = ['two_moons', 'gaussian_mixture', 'concentric_circles']
lambda_gp = 10
n_critic = 5

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed')
args = parser.parse_args()

set_seed(args.seed)

# ============================================================
# Gradient Penalty
# ============================================================
def gradient_penalty(D, real_data, fake_data, device='cpu'):
    batch_size = real_data.size(0)
    alpha = torch.rand(batch_size, *([1] * (real_data.dim() - 1)), device=device)
    interpolates = alpha * real_data + (1 - alpha) * fake_data
    interpolates.requires_grad_(True)

    d_interpolates = D(interpolates)
    fake = torch.ones_like(d_interpolates, device=device)

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.view(batch_size, -1)
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gp

# ============================================================
# WGAN-GP building blocks
# ============================================================
class Generator(nn.Module):
    def __init__(self, z_dim, hidden_dim, out_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x):
        return self.net(x)

# ============================================================
# Training
# ============================================================

datasets = TwoDDatasets(n_samples=10000).create_all_datasets()
metrics_obj = Metrics(device=device)

for dataset_name in datasets_names:
    data = datasets[dataset_name]['data']
    for config_idx, config in enumerate(configs):
        latent_dim = config['latent_dim']
        hidden_dim = config['hidden_dim']
        lr = config['lr']
        batch_size = config['batch_size']
        n_epochs = config['n_epochs']

        print(f"\n=== Dataset: {dataset_name}, Config {config_idx + 1} ===")

        data_loader = DataLoader(TensorDataset(data), batch_size=batch_size, shuffle=True, drop_last=True)

        G = Generator(latent_dim, hidden_dim).to(device)
        D = Discriminator(hidden_dim=hidden_dim).to(device)

        optimizer_G = optim.Adam(G.parameters(), lr=lr, betas=(0.0, 0.9))
        optimizer_D = optim.Adam(D.parameters(), lr=lr, betas=(0.0, 0.9))

        metrics_history = []
        start_time = time.time()

        for epoch in trange(1, n_epochs + 1):
            for i, (real_data,) in enumerate(data_loader):
                real_data = real_data.to(device)

                for _ in range(n_critic):
                    z = torch.randn(batch_size, latent_dim, device=device)
                    fake_data = G(z).detach()
                    d_real = D(real_data).mean()
                    d_fake = D(fake_data).mean()
                    gp = gradient_penalty(D, real_data, fake_data, device=device)

                    loss_D = d_fake - d_real + lambda_gp * gp
                    optimizer_D.zero_grad()
                    loss_D.backward()
                    optimizer_D.step()

                z = torch.randn(batch_size, latent_dim, device=device)
                fake_data = G(z)
                loss_G = -D(fake_data).mean()
                optimizer_G.zero_grad()
                loss_G.backward()
                optimizer_G.step()

            if epoch % 50 == 0 or epoch == 1:
                G.eval()
                with torch.no_grad():
                    z = torch.randn(1000, latent_dim, device=device)
                    gen_samples = G(z).cpu()

                metric_values = metrics_obj.compute_all_metrics(real_data.cpu(), gen_samples, true_distribution=dataset_name)

                try:
                    n_modes = len(torch.unique(datasets[dataset_name]['labels']))
                    distances = torch.cdist(gen_samples, datasets[dataset_name]['data'])
                    mode_hits = ((distances < 0.5).any(dim=0)).sum().item()
                    coverage = mode_hits / n_modes
                except:
                    coverage = 0.0

                metric_values['mode_coverage'] = coverage
                metric_values['epoch'] = epoch
                metrics_history.append(metric_values)

                # Визуализация
                plt.figure(figsize=(5,5))
                plt.scatter(real_data[:,0].cpu(), real_data[:,1].cpu(), color='blue', alpha=0.3, label='Real')
                plt.scatter(gen_samples[:,0], gen_samples[:,1], color='red', alpha=0.5, label='Generated')
                plt.title(f"{dataset_name} - Config {config_idx+1} - Epoch {epoch}")
                plt.legend()
                plt.savefig(f"{output_dir}/{dataset_name}_config{config_idx+1}_epoch{epoch}.png")
                plt.close()
                G.train()

        total_time = time.time() - start_time
        print(f"Total training time: {total_time:.2f} seconds")

        df = pd.DataFrame(metrics_history)
        df['total_training_time'] = total_time
        df.to_csv(f"{output_dir}/metrics_{dataset_name}_config{config_idx+1}.csv", index=False)

print("Все обучение завершено!")
