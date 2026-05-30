from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset


def random_subset_indices(n: int, limit: int | None, seed: int) -> torch.Tensor:
    if limit is None or limit >= n:
        return torch.arange(n)
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(n, generator=generator)[:limit]


def synthetic_target(latent: torch.Tensor) -> torch.Tensor:
    input_dim = latent.shape[1]
    target = 0.6 * torch.sin(math.pi * latent).sum(dim=1, keepdim=True)
    target = target + 0.3 * latent.pow(2).sum(dim=1, keepdim=True)
    target = target + 0.2 * (latent[:, 0:1] * latent[:, 1:2])
    return target / math.sqrt(input_dim)


def load_synthetic_vector_regression_data(
    input_dim: int = 8,
    input_scale: float = 1.0,
    n_train: int = 512,
    n_test: int = 512,
    noise_std: float = 0.1,
    train_eval_limit: int | None = None,
    batch_size: int = 64,
    eval_batch_size: int = 512,
    seed: int = 0,
    num_workers: int = 0,
    device: torch.device | None = None,
) -> dict:
    generator = torch.Generator().manual_seed(seed)
    latent_train = torch.rand(n_train, input_dim, generator=generator) * 2.0 - 1.0
    x_train = input_scale * latent_train
    y_train = synthetic_target(latent_train) + noise_std * torch.randn(n_train, 1, generator=generator)

    latent_test = torch.rand(n_test, input_dim, generator=generator) * 2.0 - 1.0
    x_test = input_scale * latent_test
    y_test = synthetic_target(latent_test)

    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)
    train_eval_dataset = maybe_subset_dataset(train_dataset, train_eval_limit, seed + 1)
    return make_data_dict(
        train_dataset,
        train_eval_dataset,
        test_dataset,
        batch_size,
        eval_batch_size,
        num_workers,
        device,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        latent_train=latent_train,
        latent_test=latent_test,
        input_scale=input_scale,
    )


def patch_sklearn_urlretrieve_user_agent() -> None:
    """Avoid occasional HTTP 403 errors from sklearn's California Housing host."""
    import shutil as _shutil
    from urllib.request import Request as _Request
    from urllib.request import urlopen as _urlopen

    import sklearn.datasets._base as _sklearn_base

    def _urlretrieve_with_user_agent(url, filename=None, reporthook=None, data=None):
        request = _Request(url, data=data, headers={"User-Agent": "Mozilla/5.0"})
        with _urlopen(request) as response:
            if filename is None:
                import tempfile as _tempfile

                handle = _tempfile.NamedTemporaryFile(delete=False)
                filename = handle.name
                handle.close()
            with open(filename, "wb") as out_file:
                _shutil.copyfileobj(response, out_file)
            return filename, response.info()

    _sklearn_base.urlretrieve = _urlretrieve_with_user_agent


def load_california_housing(
    test_size: float = 0.2,
    batch_size: int = 256,
    eval_batch_size: int = 4096,
    seed: int = 0,
    num_workers: int = 0,
    data_home: str | None = None,
    device: torch.device | None = None,
) -> dict:
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    patch_sklearn_urlretrieve_user_agent()
    dataset = fetch_california_housing(data_home=data_home)
    x_train, x_test, y_train, y_test = train_test_split(
        dataset.data,
        dataset.target,
        test_size=test_size,
        random_state=seed,
    )

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_train = x_scaler.fit_transform(x_train)
    x_test = x_scaler.transform(x_test)
    y_train = y_scaler.fit_transform(y_train.reshape(-1, 1))
    y_test = y_scaler.transform(y_test.reshape(-1, 1))

    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)
    return make_data_dict(
        train_dataset,
        train_dataset,
        test_dataset,
        batch_size,
        eval_batch_size,
        num_workers,
        device,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
    )


def load_mnist(
    train_limit: int | None = None,
    test_limit: int | None = None,
    train_eval_limit: int | None = 4000,
    batch_size: int = 128,
    eval_batch_size: int = 1024,
    data_dir: str = "./data",
    seed: int = 0,
    mean_center_only: bool = True,
    num_workers: int = 0,
    device: torch.device | None = None,
) -> dict:
    from torchvision.datasets import MNIST

    train_dataset = MNIST(root=data_dir, train=True, download=True)
    test_dataset = MNIST(root=data_dir, train=False, download=True)
    x_train = train_dataset.data.float() / 255.0
    x_test = test_dataset.data.float() / 255.0
    train_labels = train_dataset.targets.long()
    test_labels = test_dataset.targets.long()
    return prepare_image_data(
        x_train,
        train_labels,
        x_test,
        test_labels,
        num_classes=10,
        train_limit=train_limit,
        test_limit=test_limit,
        train_eval_limit=train_eval_limit,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        seed=seed,
        mean_center_only=mean_center_only,
        num_workers=num_workers,
        device=device,
        channel_first=False,
    )


def load_cifar10(
    train_limit: int | None = None,
    test_limit: int | None = None,
    train_eval_limit: int | None = 4000,
    batch_size: int = 128,
    eval_batch_size: int = 1024,
    data_dir: str = "./data",
    seed: int = 0,
    mean_center_only: bool = True,
    num_workers: int = 0,
    device: torch.device | None = None,
) -> dict:
    from torchvision.datasets import CIFAR10

    train_dataset = CIFAR10(root=data_dir, train=True, download=True)
    test_dataset = CIFAR10(root=data_dir, train=False, download=True)
    x_train = torch.from_numpy(train_dataset.data).float() / 255.0
    x_test = torch.from_numpy(test_dataset.data).float() / 255.0
    train_labels = torch.tensor(train_dataset.targets, dtype=torch.long)
    test_labels = torch.tensor(test_dataset.targets, dtype=torch.long)
    return prepare_image_data(
        x_train,
        train_labels,
        x_test,
        test_labels,
        num_classes=10,
        train_limit=train_limit,
        test_limit=test_limit,
        train_eval_limit=train_eval_limit,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        seed=seed,
        mean_center_only=mean_center_only,
        num_workers=num_workers,
        device=device,
        channel_first=True,
    )


def prepare_image_data(
    x_train: torch.Tensor,
    train_labels: torch.Tensor,
    x_test: torch.Tensor,
    test_labels: torch.Tensor,
    num_classes: int,
    train_limit: int | None,
    test_limit: int | None,
    train_eval_limit: int | None,
    batch_size: int,
    eval_batch_size: int,
    seed: int,
    mean_center_only: bool,
    num_workers: int,
    device: torch.device | None,
    channel_first: bool,
) -> dict:
    train_idx = random_subset_indices(len(x_train), train_limit, seed)
    test_idx = random_subset_indices(len(x_test), test_limit, seed + 1)
    x_train = x_train[train_idx]
    train_labels = train_labels[train_idx]
    x_test = x_test[test_idx]
    test_labels = test_labels[test_idx]

    mean = x_train.mean()
    if mean_center_only:
        x_train = x_train - mean
        x_test = x_test - mean
    else:
        std = x_train.std().clamp_min(1e-6)
        x_train = (x_train - mean) / std
        x_test = (x_test - mean) / std

    if channel_first:
        x_train = x_train.permute(0, 3, 1, 2).contiguous()
        x_test = x_test.permute(0, 3, 1, 2).contiguous()
    x_train = x_train.view(x_train.size(0), -1)
    x_test = x_test.view(x_test.size(0), -1)
    y_train = F.one_hot(train_labels, num_classes=num_classes).float()
    y_test = F.one_hot(test_labels, num_classes=num_classes).float()
    train_dataset = TensorDataset(x_train, y_train, train_labels)
    test_dataset = TensorDataset(x_test, y_test, test_labels)
    train_eval_dataset = maybe_subset_dataset(train_dataset, train_eval_limit, seed + 2)
    return make_data_dict(
        train_dataset,
        train_eval_dataset,
        test_dataset,
        batch_size,
        eval_batch_size,
        num_workers,
        device,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        train_labels=train_labels,
        test_labels=test_labels,
    )


def maybe_subset_dataset(dataset, limit: int | None, seed: int):
    if limit is None or limit >= len(dataset):
        return dataset
    indices = random_subset_indices(len(dataset), limit, seed)
    return Subset(dataset, indices.tolist())


def make_data_dict(
    train_dataset,
    train_eval_dataset,
    test_dataset,
    batch_size: int,
    eval_batch_size: int,
    num_workers: int,
    device: torch.device | None,
    **tensors,
) -> dict:
    pin_memory = device is not None and device.type == "cuda"
    return {
        "train_loader": DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory),
        "train_eval_loader": DataLoader(train_eval_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory),
        "test_loader": DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory),
        "train_size": len(train_dataset),
        "train_eval_size": len(train_eval_dataset),
        "test_size": len(test_dataset),
        **tensors,
    }


DATA_LOADERS = {
    "load_synthetic_vector_regression_data": load_synthetic_vector_regression_data,
    "load_california_housing": load_california_housing,
    "load_mnist": load_mnist,
    "load_cifar10": load_cifar10,
}


def load_task_data(config: dict, device: torch.device) -> dict:
    loader = DATA_LOADERS[config["data_loader"]]
    kwargs = dict(config["data_kwargs"])
    kwargs["device"] = device
    return loader(**kwargs)
