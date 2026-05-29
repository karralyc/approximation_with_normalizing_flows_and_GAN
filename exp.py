import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse
from sklearn.datasets import make_circles
from sklearn.cluster import KMeans

from metrics import Metrics
from real_nvp import RealNVP, set_seed


class Experiment:
    def __init__(self, output_dir='outputs/experiment', seed=42):
        self.output_dir = output_dir
        self.seed = seed
        self.metrics = Metrics(device='cpu', seed=seed)
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f'{output_dir}/samples', exist_ok=True)

    def generate_data(self, noise_level, n_samples=10000):
        data, _ = make_circles(n_samples=n_samples, noise=noise_level, factor=0.5, random_state=self.seed)
        data = data * 1.5
        return torch.tensor(data, dtype=torch.float32)

    def compute_separation(self, data):
        radii = np.linalg.norm(data.numpy(), axis=1)
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        kmeans.fit(radii.reshape(-1, 1))
        centers = sorted(kmeans.cluster_centers_.flatten())
        distance = abs(centers[1] - centers[0])
        std = radii.std()
        return min(distance / (2 * std), 1.0) if std > 0 else 0.0

    def compute_coverage(self, real_data, gen_data):
        real_r = np.linalg.norm(real_data.numpy(), axis=1)
        gen_r = np.linalg.norm(gen_data, axis=1)

        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        kmeans.fit(real_r.reshape(-1, 1))
        centers = sorted(kmeans.cluster_centers_.flatten())

        real_labels = kmeans.labels_

        inner_radii = real_r[real_labels == 0]
        outer_radii = real_r[real_labels == 1]

        if len(inner_radii) == 0 or len(outer_radii) == 0:
            return 0.5

        inner_low = np.percentile(inner_radii, 5)
        inner_high = np.percentile(inner_radii, 95)
        outer_low = np.percentile(outer_radii, 5)
        outer_high = np.percentile(outer_radii, 95)

        inner_points = np.sum((gen_r >= inner_low) & (gen_r <= inner_high))
        outer_points = np.sum((gen_r >= outer_low) & (gen_r <= outer_high))

        min_points = 200

        inner_covered = inner_points >= min_points
        outer_covered = outer_points >= min_points

        return (float(inner_covered) + float(outer_covered)) / 2.0

    def train_model(self, train_data, val_data, config):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        batch_size = 256

        model = RealNVP(dim=2, n_layers=config['n_layers'],
                        hidden_dim=config['hidden_dim'], use_batchnorm=False).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

        train_data = train_data.to(device)
        best_w1 = float('inf')
        best_samples = None

        for epoch in range(1, config['n_epochs'] + 1):
            perm = torch.randperm(train_data.size(0))
            for i in range(0, train_data.size(0), batch_size):
                batch = train_data[perm[i:i + batch_size]]
                if batch.size(0) < batch_size:
                    continue
                z, log_det = model(batch)
                log_prob = -0.5 * (z ** 2).sum(dim=1)
                loss = -(log_prob + log_det).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if epoch % 50 == 0 or epoch == config['n_epochs']:
                with torch.no_grad():
                    z = torch.randn(5000, 2, device=device)
                    samples = model.inverse(z)[0].cpu().numpy()
                w1 = self.metrics.compute_w1_distance(val_data.numpy(), samples)
                if w1 < best_w1:
                    best_w1 = w1
                    best_samples = samples

        return best_w1, best_samples

    def save_visualization(self, real_data, gen_data, noise_level, w1, coverage, output_dir):
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        axes[0].scatter(real_data[:, 0], real_data[:, 1], alpha=0.3, s=3, c='blue')
        axes[0].set_title(f'Real Data (noise={noise_level})')
        axes[0].set_aspect('equal')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('y')
        axes[0].grid(True, alpha=0.3)

        axes[1].scatter(gen_data[:, 0], gen_data[:, 1], alpha=0.3, s=3, c='red')
        axes[1].set_title(f'Generated (W1={w1:.3f}, Coverage={coverage:.1f})')
        axes[1].set_aspect('equal')
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('y')
        axes[1].grid(True, alpha=0.3)

        plt.suptitle(f'Concentric Circles | noise={noise_level}')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/noise_{noise_level}.png', dpi=150)
        plt.close()

    def run(self, noise_levels, config):
        results = []

        for idx, noise in enumerate(noise_levels):
            print(f"[{idx + 1}/{len(noise_levels)}] noise={noise:.2f}")

            data = self.generate_data(noise)
            separation = self.compute_separation(data)

            n_train = int(0.8 * len(data))
            train_data, val_data = data[:n_train], data[n_train:]

            w1, samples = self.train_model(train_data, val_data, config)

            coverage = self.compute_coverage(val_data, samples)

            self.save_visualization(val_data.numpy(), samples, noise, w1, coverage,
                                    f'{self.output_dir}/samples')

            results.append({
                'noise': noise,
                'w1': w1,
                'coverage': coverage,
                'separation': separation
            })

            print(f"  W1={w1:.4f}, Coverage={coverage:.1f}")

        return results

    def save_results(self, results, config):
        df = pd.DataFrame(results)
        df.to_csv(f'{self.output_dir}/results.csv', index=False)
        pd.DataFrame([config]).to_csv(f'{self.output_dir}/config.csv', index=False)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        noises = [r['noise'] for r in results]

        axes[0].plot(noises, [r['w1'] for r in results], 'o-', color='darkblue', linewidth=2)
        axes[0].set_xlabel('Noise Level')
        axes[0].set_ylabel('Wasserstein-1 Distance')
        axes[0].set_title('W1 vs Noise')
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(noises, [r['coverage'] for r in results], 'o-', color='purple', linewidth=2)
        axes[1].axhline(y=0.5, color='orange', linestyle='--', label='One mode')
        axes[1].axhline(y=1.0, color='green', linestyle='--', label='Both modes')
        axes[1].set_xlabel('Noise Level')
        axes[1].set_ylabel('Mode Coverage')
        axes[1].set_title('Coverage vs Noise')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(noises, [r['separation'] for r in results], 'o-', color='darkgreen', linewidth=2)
        axes[2].set_xlabel('Noise Level')
        axes[2].set_ylabel('Separation Score')
        axes[2].set_title('Ring Separation vs Noise')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/summary.png', dpi=150)
        plt.close()

        print(f"\nResults saved to {self.output_dir}")

        zero_result = next(r for r in results if r['noise'] == 0.0)
        print(f"At noise=0.0: Coverage = {zero_result['coverage']:.1f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise_levels', type=str, default='0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5')
    parser.add_argument('--n_epochs', type=int, default=150)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--n_layers', type=int, default=6)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='outputs/experiment')

    args = parser.parse_args()

    noise_levels = [float(x) for x in args.noise_levels.split(',')]

    config = {
        'n_layers': args.n_layers,
        'hidden_dim': args.hidden_dim,
        'lr': 1e-3,
        'n_epochs': args.n_epochs
    }

    set_seed(args.seed)

    exp = Experiment(output_dir=args.output_dir, seed=args.seed)
    results = exp.run(noise_levels, config)
    exp.save_results(results, config)


if __name__ == "__main__":
    main()