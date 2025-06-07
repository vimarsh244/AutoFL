import torch
import flwr
from flwr.simulation import run_simulation
from flwr.client import ClientApp
from flwr.server import ServerApp

from omegaconf import OmegaConf
import warnings

from mclientCL import client_fn
from mclserver import server_fn

# Ignore Deprecation Warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Load Config
cfg = OmegaConf.load('config/config.yaml')
print(OmegaConf.to_yaml(cfg))

def get_model():
    """Get model based on configuration"""
    if cfg.model.name == "resnet":
        from models.ResNet import ResNet
        return ResNet(num_classes=100 if cfg.dataset.workload == "cifar100" else 10)
    elif cfg.model.name == "simple_cnn":
        from models.SimpleCNN import Net
        return Net()
    else:
        raise ValueError(f"Unknown model: {cfg.model.name}")

def main():
    client = ClientApp(client_fn=client_fn)
    server = ServerApp(server_fn=server_fn)
    backend_config = {
            "client_resources": {
                "num_cpus": cfg.client.num_cpus,
                "num_gpus": cfg.client.num_gpus,
                }
            }

    # Run Simulation
    print("Running Simulation")

    run_simulation(
        server_app = server,
        client_app = client,
        num_supernodes = cfg.server.num_clients,
        backend_config = backend_config,
    )

if __name__ == "__main__":
    main()
