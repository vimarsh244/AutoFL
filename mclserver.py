import flwr
from flwr.common import Metrics, Context
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg, FedProx, FedOpt

import wandb
import os
from omegaconf import OmegaConf

from clutils.scallbacks import evaluate_metrics_aggregation_fn, fit_metrics_aggregation_fn, fit_config, eval_config

# Import custom SOTA server strategies
from algorithms.fedweit import FedWeITServerStrategy

#Setting up Configuration
from config_utils import load_config
cfg = load_config()

NUM_ROUNDS = cfg.server.num_rounds
NUM_CLIENTS = cfg.server.num_clients

# Simulation availability support ---
try:
    from sim.availability import load_scheduler_from_csvs
except Exception:
    load_scheduler_from_csvs = None

class ScheduledFedAvg(FedAvg):
    def __init__(self, config: dict):
        super().__init__(config)
        self.config = config
        self.current_round = 0
        self.sim_config = config.get('sim', {})
        
        # Initialize simulation components
        if self.sim_config.get('enabled', False):
            print("[ScheduledFedAvg] Initializing simulation components...")
            
            # Import here to avoid circular imports
            from sim.availability import create_enhanced_simulator
            
            try:
                self.availability_scheduler = create_enhanced_simulator(self.sim_config)
                if self.availability_scheduler:
                    print(f"[ScheduledFedAvg] ✅ Availability scheduler initialized with {len(self.availability_scheduler.client_intervals_s)} clients")
                else:
                    print("[ScheduledFedAvg] ⚠️ Failed to create availability scheduler - using all clients")
                    self.availability_scheduler = None
            except Exception as e:
                print(f"[ScheduledFedAvg] ❌ Error creating availability scheduler: {e}")
                self.availability_scheduler = None
        else:
            self.availability_scheduler = None
            print("[ScheduledFedAvg] Simulation disabled - using all clients")


def create_server_strategy():
    """Create and return the appropriate server strategy based on configuration."""
    
    # Get strategy name from config
    strategy_name = getattr(cfg.server, 'strategy', 'fedavg').lower()
    print(f"[Server] Using server strategy: {strategy_name}")
    
    # Common parameters for standard Flower strategies
    common_params = {
        'fraction_fit': cfg.server.fraction_fit,
        'fraction_evaluate': cfg.server.fraction_eval,
        'min_fit_clients': cfg.server.min_fit,
        'min_evaluate_clients': cfg.server.min_eval,
        'min_available_clients': cfg.server.num_clients,
        'on_fit_config_fn': fit_config,
        'on_evaluate_config_fn': eval_config,
        'evaluate_metrics_aggregation_fn': evaluate_metrics_aggregation_fn,
        'fit_metrics_aggregation_fn': fit_metrics_aggregation_fn
    }

    # Build scheduler if enabled
    scheduler = None
    sim_cfg = getattr(cfg, 'sim', None)
    if sim_cfg and getattr(sim_cfg, 'enabled', False) and load_scheduler_from_csvs is not None:
        csvs = [str(p) for p in getattr(sim_cfg, 'csv_files', [])]
        try:
            scheduler = load_scheduler_from_csvs(
                csv_paths=csvs,
                signal=getattr(sim_cfg, 'signal', 'transmissionState'),
                threshold=getattr(sim_cfg, 'threshold', None),
                time_scale=getattr(sim_cfg, 'time_scale', 1.0),
                round_duration_s=getattr(sim_cfg, 'round_duration_s', 1.0),
                start_time_s=getattr(sim_cfg, 'start_time_s', 0.0),
                randomize_mapping=getattr(sim_cfg, 'randomize_mapping', False),
                num_clients=cfg.server.num_clients,
                rnd_seed=getattr(sim_cfg, 'seed', 42),
                min_gap_s=getattr(sim_cfg, 'min_gap_s', 0.0),
            )
            print("[Server] Availability scheduler loaded")
        except Exception as e:
            print(f"[Server] Failed to load scheduler: {e}")
            scheduler = None

    # Standard Flower strategies
    if strategy_name == 'fedavg':
        if scheduler:
            return ScheduledFedAvg(scheduler=scheduler, sim_config=sim_cfg, **common_params)
        return FedAvg(**common_params)
    
    elif strategy_name == 'fedprox':
        # FedProx-specific parameters
        proximal_mu = getattr(cfg.server, 'fedprox', {}).get('mu', 0.01)
        if scheduler:
            return ScheduledFedAvg(scheduler=scheduler, sim_config=sim_cfg, proximal_mu=proximal_mu, **common_params)
        return FedProx(proximal_mu=proximal_mu, **common_params)
    
    # Note: Scaffold and FedNova not available in current Flower version
    elif strategy_name in ['scaffold', 'fednova']:
        print(f"[Server] {strategy_name} not available in current Flower version, using FedAvg")
        if scheduler:
            return ScheduledFedAvg(scheduler=scheduler, sim_config=sim_cfg, **common_params)
        return FedAvg(**common_params)
    
    elif strategy_name == 'fedopt':
        # FedOpt-specific parameters
        server_optimizer = getattr(cfg.server, 'fedopt', {}).get('server_optimizer', 'adam')
        server_lr = getattr(cfg.server, 'fedopt', {}).get('server_lr', 1.0)
        beta1 = getattr(cfg.server, 'fedopt', {}).get('beta1', 0.9)
        beta2 = getattr(cfg.server, 'fedopt', {}).get('beta2', 0.999)
        if scheduler:
            print("[Server] Scheduler active with FedOpt: filtering not applied; consider FedAvg/FedProx for filtering")
        return FedOpt(
            server_optimizer=server_optimizer,
            server_lr=server_lr,
            beta1=beta1,
            beta2=beta2,
            **common_params
        )
    
    # Custom SOTA strategies
    elif strategy_name == 'fedweit':
        print("[Server] FedWeIT strategy selected")
        print("[Server] Note: Using FedAvg for now. Full FedWeIT server aggregation requires custom Flower strategy implementation.")
        print("[Server] Custom FedWeIT aggregation logic is available in algorithms/fedweit.py")
        if scheduler:
            return ScheduledFedAvg(scheduler=scheduler, sim_config=sim_cfg, **common_params)
        return FedAvg(**common_params)
    
    # For other SOTA strategies that don't need custom server aggregation
    elif strategy_name in ['plora', 'fedcprompt', 'fedet', 'fedgem', 'fedma', 'fedproto', 'fedrcil', 'fedrep', 'sacfl', 'stamp']:
        print(f"[Server] Using FedAvg for SOTA strategy: {strategy_name}")
        print(f"[Server] Note: {strategy_name} uses custom client logic but standard server aggregation")
        if scheduler:
            return ScheduledFedAvg(scheduler=scheduler, sim_config=sim_cfg, **common_params)
        return FedAvg(**common_params)
    
    else:
        print(f"[Server] Warning: Unknown strategy '{strategy_name}', falling back to FedAvg")
        if scheduler:
            return ScheduledFedAvg(scheduler=scheduler, sim_config=sim_cfg, **common_params)
        return FedAvg(**common_params)

# Create the appropriate strategy
strategy = create_server_strategy()

def server_fn(context: Context) -> ServerAppComponents:
    """Construct components that set the ServerApp behaviour.

    You can use the settings in `context.run_config` to parameterize the
    construction of all elements (e.g the strategy or the number of rounds)
    wrapped in the returned ServerAppComponents object.
    """

    # Configure the server for specified number of rounds
    config = ServerConfig(cfg.server.num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


