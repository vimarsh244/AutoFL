# Using Domain Incremental Workloads with Hydra

This guide explains how to use the domain incremental workloads with Hydra configuration system.

## Configuration Structure

The configuration is organized in the following structure:
```
config/
├── config.yaml          # Main configuration file
├── workload/            # Workload-specific configurations
│   ├── cifar10.yaml
│   ├── cifar100.yaml
│   ├── bdd100k.yaml
│   └── kitti.yaml
└── model/              # Model configurations
    ├── resnet.yaml
    └── simple_cnn.yaml
```

## Example Configuration Files

### 1. Main Configuration (config.yaml)
```yaml
defaults:
  - workload: cifar10  # Choose workload: cifar10, cifar100, bdd100k, kitti
  - model: resnet      # Choose model architecture
  - _self_

server:
  num_rounds: 10
  num_clients: 10
  fraction_fit: 0.8
  fraction_eval: 0.2
  min_fit: 8
  min_eval: 2

training:
  batch_size: 32
  learning_rate: 0.001
  epochs: 5
  optimizer: adam

logging:
  wandb:
    project: "domain-incremental-fl"
    entity: null
    tags: []
```

### 2. Workload Configurations

#### CIFAR10 (config/workload/cifar10.yaml)
```yaml
workload:
  name: cifar10
  domains:
    - original
    - brightness
    - contrast
    - blur
  experiences_per_domain: 2
  transform_params:
    brightness: 0.5
    contrast: 0.5
    blur_kernel: 3
```

#### BDD100K (config/workload/bdd100k.yaml)
```yaml
workload:
  name: bdd100k
  domains:
    weather:
      - clear
      - cloudy
      - rainy
      - snowy
    timeofday:
      - daytime
      - night
  experiences_per_domain: 2
  image_size: [224, 224]
```

### 3. Model Configuration (config/model/resnet.yaml)
```yaml
model:
  name: resnet18
  pretrained: true
  num_classes: 10  # Will be overridden by workload
  freeze_backbone: false
```

## Running Experiments

### 1. Basic Run
```bash
python mclmain.py workload=cifar10 model=resnet
```

### 2. Override Configuration
```bash
python mclmain.py workload=cifar10 model=resnet training.batch_size=64 server.num_rounds=20
```

### 3. Multi-run (Grid Search)
```bash
python mclmain.py -m workload=cifar10,cifar100 training.learning_rate=0.001,0.0001
```

## Code Integration

### 1. Main Script (mclmain.py)
```python
import hydra
from omegaconf import DictConfig, OmegaConf
from workloads import get_workload

@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):
    # Initialize workload
    workload = get_workload(cfg.workload.name)
    
    # Initialize model
    model = get_model(cfg.model)
    
    # Start federated learning
    start_federated_learning(cfg, workload, model)

if __name__ == "__main__":
    main()
```

### 2. Workload Factory (workloads/__init__.py)
```python
from .CIFAR10DomainCL import get_dataloaders as cifar10_dataloaders
from .CIFAR100DomainCL import get_dataloaders as cifar100_dataloaders
from .BDD100KDomainCL import get_dataloaders as bdd100k_dataloaders
from .KITTIDomainCL import get_dataloaders as kitti_dataloaders

WORKLOAD_MAP = {
    "cifar10": cifar10_dataloaders,
    "cifar100": cifar100_dataloaders,
    "bdd100k": bdd100k_dataloaders,
    "kitti": kitti_dataloaders
}

def get_workload(workload_name: str):
    if workload_name not in WORKLOAD_MAP:
        raise ValueError(f"Unknown workload: {workload_name}")
    return WORKLOAD_MAP[workload_name]
```

## Example Usage Scenarios

### 1. Running CIFAR10 with Different Models
```bash
# Run with ResNet
python mclmain.py workload=cifar10 model=resnet

# Run with Simple CNN
python mclmain.py workload=cifar10 model=simple_cnn
```

### 2. Experimenting with Different Domain Combinations
```bash
# Run BDD100K with only daytime data
python mclmain.py workload=bdd100k workload.domains.timeofday=[daytime]

# Run KITTI with only urban roads
python mclmain.py workload=kitti workload.domains.road_type=[urban]
```

### 3. Hyperparameter Tuning
```bash
# Grid search over learning rates and batch sizes
python mclmain.py -m \
    workload=cifar10 \
    training.learning_rate=0.001,0.0001 \
    training.batch_size=32,64,128
```

## Best Practices

1. **Configuration Organization**
   - Keep workload-specific configs in `config/workload/`
   - Keep model-specific configs in `config/model/`
   - Use defaults in main config.yaml

2. **Experiment Tracking**
   - Use Hydra's output directory for experiment tracking
   - Enable wandb logging for visualization
   - Save configuration with each run

3. **Multi-run Experiments**
   - Use Hydra's multi-run feature for hyperparameter tuning
   - Organize results by workload and model type
   - Use tags for better experiment organization

## Troubleshooting

1. **Configuration Errors**
   - Check config file syntax
   - Verify all required fields are present
   - Use `--info` flag for detailed config info

2. **Workload-Specific Issues**
   - Verify dataset paths
   - Check domain attribute names
   - Monitor memory usage

3. **Hydra-Specific Issues**
   - Use `hydra.verbose=true` for debugging
   - Check output directory permissions
   - Verify config file locations 