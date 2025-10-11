# AutoFL

A framework for federated continual learning that combines Flower (federated learning) with Avalanche (continual learning). Built for experimenting with autonomous vehicles data scenarios like non-IID data, domain shifts, and network latency constraints.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Server (Flower)                         │
│  - Aggregation strategies (FedAvg, Latency-aware)              │
│  - Round coordination                                            │
│  - Metrics collection                                            │
└─────────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
┌───────▼───────┐   ┌───────▼───────┐   ┌───────▼───────┐
│   Client 1    │   │   Client 2    │   │   Client N    │
│ ┌───────────┐ │   │ ┌───────────┐ │   │ ┌───────────┐ │
│ │ Avalanche │ │   │ │ Avalanche │ │   │ │ Avalanche │ │
│ │ CL Engine │ │   │ │ CL Engine │ │   │ │ CL Engine │ │
│ └───────────┘ │   │ └───────────┘ │   │ └───────────┘ │
│               │   │               │   │               │
│ - Local data  │   │ - Local data  │   │ - Local data  │
│ - Experiences │   │ - Experiences │   │ - Experiences │
│ - Replay buf  │   │ - Replay buf  │   │ - Replay buf  │
└───────────────┘   └───────────────┘   └───────────────┘

Data Flow:
1. Server broadcasts model to clients
2. Clients train on sequential experiences (continual learning)
3. Clients send updated weights back to server
4. Server aggregates weights (federated learning)
5. Repeat for multiple rounds
```

## Quick Start

```bash
# Run with default settings (CIFAR10, SimpleCNN, 10 clients)
python mclmain.py

# Use experiment presets
python mclmain.py --config-path config/experiments --config-name cifar10_naive
python mclmain.py --config-path config/experiments --config-name cifar100_resnet

# Override parameters
python mclmain.py dataset.workload=cifar100 model.name=resnet server.num_rounds=20
```

## Supported Workloads

### Standard Benchmarks
- **CIFAR10/100**: 10 or 100 class image classification
- **Permuted MNIST**: Pixel permutation tasks
- **Split CIFAR**: Class-incremental learning (2-5 classes per task)

### Driving Datasets
- **BDD100K** (10k subset): Weather and time-of-day domain shifts

Domain splits create multiple experiences per client. For example, with CIFAR10:
- Domain 1: Original images
- Domain 2: Brightness adjusted
- Domain 3: Contrast adjusted  
- Domain 4: Gaussian blur applied

Each domain is further split into 2 experiences, giving 8 sequential learning tasks per client.

## Model Architectures

Supported model architectures for image clasification task:
- SimpleCNN
- Resnet (18, 34, 50)
- MobileNet V2 & V3 (small & large)


## Continual Learning Strategies

**Naive**: Standard incremental training (prone to catastrophic forgetting)

**Domain**: Learn from domain-shifted data (brightness, contrast, blur transformations)

**Experience Replay**: Store samples from old tasks in a buffer and replay during new task training
```yaml
cl:
  strategy: replay
  replay_mem_size: 200
  replay_selection: random  # or herding, closest_to_mean
```

**Elastic Weight Consolidation (EWC)**: Protect important weights from being overwritten
```yaml
cl:
  strategy: ewc
  ewc_lambda: 0.4
```

**Hybrid**: Combines EWC + replay for best anti-forgetting performance
```yaml
cl:
  strategy: hybrid
  ewc_lambda: 0.4
  replay_mem_size: 300
```

## Latency Simulation

Inject realistic network delays from OMNeT++ vehicular traces:

```yaml
latency:
  enabled: true
  csv_path: omnet-data/latency_with_10cars2RSU_30.09.2025.csv
  sampling_mode: chunk  # mean, trace, chunk, random
  scaling_factor: 1.0
  threshold_multiplier: 10000.0
  drop_behavior: skip  # skip or remove slow clients
```

**Sampling modes:**
- `mean`: Use average delay per client
- `trace`: Replay exact time series
- `chunk`: Split trace into round-sized windows
- `random`: Random sampling from distribution

## Non-IID Data Distribution

Simulate heterogeneous federated settings:

```yaml
dataset:
  split: niid
  niid:
    alpha: 0.5  # Lower = more heterogeneous (Dirichlet concentration)
    min_samples: 10
```

For label-based heterogeneity:
```yaml
dataset:
  split: niid_label
  niid:
    classes_per_client: 2  # Each client sees only 2 classes
```

## Configuration

Experiments use YAML configs. Base config at `config/config.yaml`, experiment presets in `config/experiments/`.

Key sections:

```yaml
server:
  num_rounds: 10
  num_clients: 10
  strategy: latency_aware_fedavg

client:
  epochs: 3
  num_gpus: 0.2  # Fractional GPU allocation

model:
  name: simple_cnn
  num_classes: 10

dataset:
  workload: cifar10
  batch_size: 32
  split: iid

cl:
  strategy: naive
  num_experiences: 5

training:
  learning_rate: 0.001
  optimizer: adam

wb:
  project: autofl-testing
  mode: online
```

## Outputs and Logging

Each run creates a timestamped directory under `outputs/` with:
- `client_round_metrics.csv`: Per-client timing, latency, participation
- `server_round_metrics.csv`: Aggregation time, dropout info
- `aggregate_metrics.csv`: Accuracy, forgetting, timing per round
- WandB logs (if enabled)

## GPU Usage

GPU allocation is automatic when available:
```yaml
client:
  num_gpus: 0.2  # 5 clients share 1 GPU
```

Ray handles fractional GPU sharing across clients. Models and data automatically move to CUDA when detected.

## Example Experiments

```bash
# Domain incremental learning
python mclmain.py --config-path config/experiments --config-name cifar10_domain

# Experience replay strategy
python mclmain.py --config-path config/experiments --config-name split_cifar10_replay

# EWC with permuted MNIST
python mclmain.py --config-path config/experiments --config-name permuted_mnist_ewc

```

## Project Structure

```
mclmain.py              # Main entry point
mclientCL.py            # Client initialization
mclserver.py            # Server setup and strategy resolution
config/                 # YAML configurations
  experiments/          # Preset experiment configs
workloads/              # Dataset loaders (CIFAR, BDD100K, etc)
models/                 # Network architectures
algorithms/             # FL strategies (FedAvg, latency-aware)
clutils/                # CL strategy builders (EWC, replay)
clients/                # Flower client implementation
utils/                  # Latency simulation, metrics
omnet-data/             # OMNeT++ traces and analysis scripts
```

## Development

See `tasks.md` for current development status and roadmap.
