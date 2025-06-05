NUM_CLIENTS = cfg.server.num_clients
BATCH_SIZE = 32

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from avalanche.benchmarks.utils import as_classification_dataset, AvalancheDataset, as_avalanche_dataset
from avalanche.benchmarks.utils.data import make_avalanche_dataset

import flwr
from flwr_datasets import FederatedDataset

from omegaconf import OmegaConf

# Setup Config
cfg = OmegaConf.load('config/config.yaml')

NUM_CLIENTS = cfg.server.num_clients
BATCH_SIZE = cfg.dataset.batch_size

class TupleDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        return sample["img"], sample["label"]


 
def load_datasets(partition_id: int):
    fds = FederatedDataset(dataset="cifar10", partitioners={"train": NUM_CLIENTS})
    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    pytorch_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    def apply_transforms(batch):
        # Instead of passing transforms to CIFAR10(..., transform=transform)
        # we will use this function to dataset.with_transform(apply_transforms)
        # The transforms object is exactly the same
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    # Create train/val for each partition and wrap it into DataLoader
    partition_train_test = partition_train_test.with_transform(apply_transforms)

    # Wrap in Avalanche Dataset

    train_CIFAR = make_avalanche_dataset(partition_train_test["train"])

    test_CIFAR = make_avalanche_dataset(partition_train_test["test"])

    trainloader = DataLoader(
        partition_train_test["train"], batch_size=BATCH_SIZE, shuffle=True
    )
    valloader = DataLoader(partition_train_test["test"], batch_size=BATCH_SIZE)
    testset = fds.load_split("test").with_transform(apply_transforms)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE)
    return TupleDataset(partition_train_test["train"]), TupleDataset(partition_train_test["test"])
#    return train_CIFAR, test_CIFAR
