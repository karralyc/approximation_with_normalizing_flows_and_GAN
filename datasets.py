import torch
import numpy as np
from sklearn.datasets import make_moons, make_circles
from torch.utils.data import DataLoader, TensorDataset, random_split
from typing import Tuple, Dict


class TwoDDatasets:

    def __init__(self, n_samples: int = 10000, random_state: int = 42):
        self.n_samples = n_samples
        self.random_state = random_state
        np.random.seed(random_state)
        torch.manual_seed(random_state)

    def create_gaussian_mixture(self, n_components: int = 8, std: float = 0.1,
                                radius: float = 2.0, normalize: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:

        angles = np.linspace(0, 2 * np.pi, n_components, endpoint=False)
        centers = np.column_stack([np.cos(angles), np.sin(angles)]) * radius

        data = []
        labels = []
        samples_per_component = self.n_samples // n_components

        for i, center in enumerate(centers):
            component_data = np.random.randn(samples_per_component, 2) * std + center
            data.append(component_data)
            labels.extend([i] * samples_per_component)

        remaining = self.n_samples - samples_per_component * n_components
        if remaining > 0:
            extra_data = np.random.randn(remaining, 2) * std + centers[0]
            data.append(extra_data)
            labels.extend([0] * remaining)

        data = np.vstack(data)
        labels = np.array(labels)

        indices = np.random.permutation(len(data))
        data = data[indices]
        labels = labels[indices]

        if normalize:
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)
            std = np.where(std == 0, 1.0, std)
            data = (data - mean) / std

        return torch.tensor(data, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

    def create_two_moons(self, noise: float = 0.05, normalize: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:

        data, labels = make_moons(
            n_samples=self.n_samples,
            noise=noise,
            random_state=self.random_state
        )

        indices = np.random.permutation(len(data))
        data = data[indices]
        labels = labels[indices]

        if normalize:
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)
            std = np.where(std == 0, 1.0, std)
            data = (data - mean) / std

        return torch.tensor(data, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

    def create_concentric_circles(self, noise: float = 0.05, factor: float = 0.5,
                                  normalize: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:

        data, labels = make_circles(
            n_samples=self.n_samples,
            noise=noise,
            factor=factor,
            random_state=self.random_state
        )

        indices = np.random.permutation(len(data))
        data = data[indices]
        labels = labels[indices]

        if normalize:
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)
            std = np.where(std == 0, 1.0, std)
            data = (data - mean) / std

        return torch.tensor(data, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

    def create_all_datasets(self, normalize: bool = True) -> Dict[str, Dict]:

        datasets = {}

        data, labels = self.create_gaussian_mixture(n_components=8, normalize=normalize)
        datasets['gaussian_mixture'] = {'data': data, 'labels': labels}

        data, labels = self.create_two_moons(normalize=normalize)
        datasets['two_moons'] = {'data': data, 'labels': labels}

        data, labels = self.create_concentric_circles(normalize=normalize)
        datasets['concentric_circles'] = {'data': data, 'labels': labels}

        return datasets

    def create_dataloader(self, data: torch.Tensor, batch_size: int = 128,
                          train_ratio: float = 0.8, shuffle: bool = True) -> Tuple[DataLoader, DataLoader]:
        dataset = TensorDataset(data)

        train_size = int(train_ratio * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=True
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False
        )

        return train_loader, test_loader