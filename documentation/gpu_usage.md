# GPU Usage in AutoFL

## Overview
AutoFL automatically detects and uses available GPUs for training. The system uses Ray for distributed computing, which handles GPU allocation across federated clients.

## GPU Detection
- Models and data are automatically moved to GPU when available
- Device selection: `torch.device("cuda:0" if torch.cuda.is_available() else "cpu")`
- Each client can use a fractional GPU allocation

## Configuration

### GPU Allocation per Client
In your experiment config, set the GPU fraction:

```yaml
client:
  num_gpus: 0.2  # 20% of GPU per client (5 clients can share 1 GPU)
  # or
  num_gpus: 1.0  # Full GPU per client (Ray will manage sharing)
```

### Optimized Settings for GPU

1. **Increase batch size** - GPUs handle larger batches efficiently:
   ```yaml
   dataset:
     batch_size: 64  # or 128 for smaller models
   ```

2. **Use more powerful models** - Better GPU utilization:
   ```yaml
   model:
     name: resnet  # Instead of simple_cnn
   ```

3. **Adjust client count** - Balance GPU memory:
   ```yaml
   server:
     num_clients: 2  # Fewer clients = more GPU memory per client
   ```

## Example Configurations

### Fast GPU Training (CIFAR10)
```bash
python mclmain.py --config-path config/experiments --config-name cifar10_gpu_fast
```

### BDD100K with GPU
```bash
python mclmain.py --config-path config/experiments --config-name bdd100k_10k_gpu
```

## Monitoring GPU Usage
```bash
# Watch GPU utilization in real-time
watch -n 1 nvidia-smi

# Check if clients are using GPU
nvidia-smi | grep python
```