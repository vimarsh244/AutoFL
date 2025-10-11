# Development Tasks

## Phase 1: Core FL+CL Integration

- [x] Implement FL with Flower
- [x] Setup Individual client CL with Avalanche
- [x] Integrate Avalanche with Flower
- [x] Basic Tests to ensure successful integration

## Phase 2: Benchmarks & Extensions

### Workloads
- [x] CIFAR10/100 domain incremental
- [x] BDD100K domain incremental (small subset working)
- [ ] KITTI domain incremental (dataset too large, needs verification)
- [x] CORe50 benchmark
- [x] Permuted/Rotated MNIST
- [x] Split CIFAR10/100

### Models
- [x] SimpleCNN
- [x] ResNet
- [x] MobileNet (v2, v3_small, v3_large)
- [x] WideResNet

### FL Strategies
- [x] FedAvg
- [x] Latency-aware FedAvg
- [ ] Additional aggregation strategies

### CL Strategies
- [x] Naive incremental
- [x] Domain incremental
- [x] Experience Replay (buffer-based)
- [x] Elastic Weight Consolidation (EWC)
- [x] Hybrid (EWC + Replay)

### Data Distribution
- [x] IID partitioning
- [x] Non-IID with Dirichlet distribution

## Phase 3: Evaluation & Monitoring

- [x] Basic CL metrics (accuracy, forgetting)
- [x] FL round metrics
- [x] WandB integration
- [x] Local backup logging

## Phase 4: Configuration Management

- [x] YAML-based configuration
- [x] Experiment presets
- [ ] Full Hydra integration

## Phase 5: Latency Simulation

- [x] OMNeT++ trace integration
- [x] Multiple sampling modes (mean, trace, chunk, random)
- [x] Latency-aware dropout
- [x] Throughput modeling
- [ ] Online learning with OMNeT++ simulation in parallel

## Phase 6: Documentation

- [x] Quickstart guide
- [x] Domain incremental workloads
- [x] Latency simulation
- [ ] API documentation
- [ ] Tutorial notebooks
