import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import flwr
from flwr_datasets import FederatedDataset
from avalanche.benchmarks import benchmark_from_datasets
from avalanche.benchmarks.utils import as_classification_dataset, AvalancheDataset, as_avalanche_dataset
from avalanche.benchmarks.utils.data import make_avalanche_dataset
from clutils.make_experiences import split_dataset

from omegaconf import OmegaConf
from pathlib import Path

# Setup Config
config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'
cfg = OmegaConf.load(config_path)

NUM_CLIENTS = cfg.server.num_clients
BATCH_SIZE = cfg.dataset.batch_size

# Define different domain transformations
DOMAIN_TRANSFORMS = {
    'original': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
    'brightness': transforms.Compose([
        transforms.ToTensor(),
        transforms.ColorJitter(brightness=0.5),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
    'contrast': transforms.Compose([
        transforms.ToTensor(),
        transforms.ColorJitter(contrast=0.5),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
    'blur': transforms.Compose([
        transforms.ToTensor(),
        transforms.GaussianBlur(kernel_size=3),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
}

def apply_transforms(batch, transform):
    batch["img"] = [transform(img) for img in batch["img"]]
    return batch

def load_datasets(partition_id: int):
    fds = FederatedDataset(dataset="cifar100", partitioners={"train": NUM_CLIENTS})
    partition = fds.load_partition(partition_id)
    
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    
    # Create experiences for each domain
    train_experiences = []
    for domain_name, transform in DOMAIN_TRANSFORMS.items():
        # Apply domain-specific transform
        domain_data = partition_train_test["train"].with_transform(
            lambda x: apply_transforms(x, transform)
        )
        # Convert to AvalancheDataset
        domain_data = make_avalanche_dataset(domain_data)
        # Split into experiences
        domain_experiences = split_dataset(domain_data, n_experiences=2)  # Split each domain into 2 experiences
        train_experiences.extend(domain_experiences)
    
    # Create test experiences
    test_experiences = []
    testset = fds.load_split("test")
    for domain_name, transform in DOMAIN_TRANSFORMS.items():
        domain_test = testset.with_transform(
            lambda x: apply_transforms(x, transform)
        )
        # Convert to AvalancheDataset
        domain_test = make_avalanche_dataset(domain_test)
        test_experiences.append(domain_test)
    
    # Create benchmark
    benchmark = benchmark_from_datasets(
        train_datasets=train_experiences,
        test_datasets=test_experiences
    )
    
    return benchmark

def get_dataloaders(partition_id: int):
    benchmark = load_datasets(partition_id)
    
    # Create dataloaders for each experience
    train_loaders = []
    for exp in benchmark.train_stream:
        train_loaders.append(
            DataLoader(exp.dataset, batch_size=BATCH_SIZE, shuffle=True)
        )
    
    test_loaders = []
    for exp in benchmark.test_stream:
        test_loaders.append(
            DataLoader(exp.dataset, batch_size=BATCH_SIZE)
        )
    
    return train_loaders, test_loaders 