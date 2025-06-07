# Domain Incremental Workloads Documentation

This document describes the domain incremental workloads implemented for Continual Federated Learning (CFL) experiments. These workloads are designed to test the model's ability to learn and adapt to different domains while maintaining privacy through federated learning.

## Overview

The workloads are implemented for four datasets:
1. CIFAR10
2. CIFAR100
3. BDD100K
4. KITTI

Each workload creates different domains through either:
- Image transformations (for CIFAR datasets)
- Natural domain splits based on dataset attributes (for BDD100K and KITTI)

## Workload Details

### CIFAR10 and CIFAR100 DomainCL

These workloads create domains through different image transformations:

1. **Original Domain**
   - Basic normalization
   - Serves as baseline domain

2. **Brightness Domain**
   - Applies brightness variation
   - Tests model's robustness to lighting changes

3. **Contrast Domain**
   - Applies contrast variation
   - Tests model's ability to handle different contrast levels

4. **Blur Domain**
   - Applies Gaussian blur
   - Tests model's robustness to image sharpness variations

Each domain is split into 2 experiences for training, creating a total of 8 experiences per client.

### BDD100K DomainCL

Creates domains based on weather conditions and time of day:

1. **Clear Weather**
   - Daytime
   - Nighttime

2. **Cloudy Weather**
   - Daytime
   - Nighttime

3. **Rainy Weather**
   - Daytime
   - Nighttime

4. **Snowy Weather**
   - Daytime
   - Nighttime

Each domain combination is split into 2 experiences, creating multiple experiences per client depending on available data.

### KITTI DomainCL

Creates domains based on weather conditions and road types:

1. **Sunny Weather**
   - Urban roads
   - Highway
   - Residential areas

2. **Cloudy Weather**
   - Urban roads
   - Highway
   - Residential areas

3. **Rainy Weather**
   - Urban roads
   - Highway
   - Residential areas

Each domain combination is split into 2 experiences, creating multiple experiences per client depending on available data.

## Usage Instructions

### Prerequisites

1. Install required packages:
```bash
pip install torch torchvision flwr flwr-datasets avalanche-lib
```

2. Ensure you have access to the datasets:
   - CIFAR10/100: Automatically downloaded through torchvision
   - BDD100K: Download from [BDD100K website](https://bdd-data.berkeley.edu/)
   - KITTI: Download from [KITTI website](http://www.cvlibs.net/datasets/kitti/)

### Running Experiments

1. **Basic Usage**
```python
from workloads.CIFAR10DomainCL import get_dataloaders

# Get dataloaders for client 0
train_loaders, test_loaders = get_dataloaders(partition_id=0)
```

2. **Federated Learning Setup**
```python
import flwr as fl
from workloads.CIFAR10DomainCL import get_dataloaders

# Define client
class DomainClient(fl.client.NumPyClient):
    def __init__(self, partition_id):
        self.train_loaders, self.test_loaders = get_dataloaders(partition_id)
        # Initialize model and other components...

# Start client
fl.client.start_numpy_client(
    server_address="[::]:8080",
    client=DomainClient(partition_id=0)
)
```

### Configuration

Each workload can be configured through the following parameters:

- `NUM_CLIENTS`: Number of federated clients (default: 10)
- `BATCH_SIZE`: Batch size for training (default: 32)
- Number of experiences per domain (currently set to 2)

## Implementation Details

### Data Partitioning

1. Each dataset is partitioned among clients using `FederatedDataset`
2. Each client's data is split into train/test (80/20)
3. Domains are created either through:
   - Transformations (CIFAR datasets)
   - Attribute filtering (BDD100K, KITTI)
4. Each domain is split into multiple experiences
5. Test sets are also divided into domains

### Continual Learning Integration

The workloads use Avalanche's benchmark system for continual learning:
- `benchmark_from_datasets`: Creates a benchmark from domain-specific datasets
- `train_stream`: Iterator over training experiences
- `test_stream`: Iterator over test experiences

## Notes on FCL Flows

The workloads support two types of Federated Continual Learning (FCL) flows:

1. **Flow 1**: Each client trains continually on all its experiences each round
   - Use `clmain.py`
   - All experiences are trained in each round

2. **Flow 2**: Each client trains on one experience sequentially in each round
   - Use `mclmain.py`
   - One experience per round
   - Experiences are trained sequentially

## Best Practices

1. **Data Distribution**
   - Ensure balanced distribution of domains across clients
   - Monitor domain distribution in each client's data

2. **Model Selection**
   - Use models that can handle domain shifts
   - Consider using domain adaptation techniques

3. **Evaluation**
   - Evaluate on all domains regularly
   - Track performance per domain
   - Monitor forgetting across domains

4. **Hyperparameter Tuning**
   - Adjust learning rates for different domains
   - Consider domain-specific optimizers
   - Tune batch sizes based on domain complexity

## Troubleshooting

1. **Memory Issues**
   - Reduce batch size
   - Use gradient accumulation
   - Implement data streaming

2. **Training Stability**
   - Use learning rate scheduling
   - Implement gradient clipping
   - Monitor loss curves per domain

3. **Data Loading**
   - Ensure proper data paths
   - Check data format compatibility
   - Verify attribute names for BDD100K and KITTI

## Future Improvements

1. **Domain Creation**
   - Add more transformation types
   - Implement domain mixing
   - Add domain difficulty levels

2. **Evaluation Metrics**
   - Add domain-specific metrics
   - Implement forgetting measures
   - Add transfer learning metrics

3. **Data Management**
   - Implement data streaming
   - Add data augmentation
   - Implement domain balancing

## References

1. CIFAR10/100: [CIFAR Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
2. BDD100K: [BDD100K Dataset](https://bdd-data.berkeley.edu/)
3. KITTI: [KITTI Dataset](http://www.cvlibs.net/datasets/kitti/)
4. Avalanche: [Avalanche Documentation](https://avalanche.continualai.org/)
5. Flower: [Flower Documentation](https://flower.dev/) 