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
    def __init__(self, scheduler, sim_config, **kwargs):
        super().__init__(**kwargs)
        self.availability_scheduler = scheduler
        self.sim_config = sim_config

    def _filter_by_availability(self, items, round_idx: int):
        if not items:
            return items
        available = set(self.availability_scheduler.get_available_clients(round_idx))
        # Detect shape: list of ClientProxy vs list of (ClientProxy, Ins)
        first = items[0]
        if hasattr(first, 'cid'):
            filtered = [c for c in items if int(c.cid) in available]
        elif isinstance(first, tuple) and len(first) == 2 and hasattr(first[0], 'cid'):
            filtered = [(c, ins) for (c, ins) in items if int(c.cid) in available]
        else:
            # Unknown shape, do not filter
            return items
        return filtered

    def configure_fit(self, server_round: int, parameters, client_manager):
        """Configure which clients to use for training in this round based on availability."""
        # First get the base selection from the parent strategy
        base_instructions = super().configure_fit(server_round, parameters, client_manager)
        
        if not self.availability_scheduler or not self.sim_config.get("enabled", False):
            return base_instructions
        
        # Filter based on availability
        if base_instructions:
            filtered_instructions = self._filter_by_availability(base_instructions, server_round)
            if filtered_instructions != base_instructions:
                print(f"[Server] Round {server_round}: Filtered {len(base_instructions)} clients to {len(filtered_instructions)} available clients")
            return filtered_instructions
        return base_instructions

    def configure_evaluate(self, server_round, parameters, client_manager):
        """Configure which clients to use for evaluation in this round based on availability."""
        # First get the base selection from the parent strategy
        base_instructions = super().configure_evaluate(server_round, parameters, client_manager)
        
        if not self.availability_scheduler or not self.sim_config.get("enabled", False):
            return base_instructions
        
        # Filter based on availability
        if base_instructions:
            filtered_instructions = self._filter_by_availability(base_instructions, server_round)
            if filtered_instructions != base_instructions:
                print(f"[Server] Round {server_round}: Filtered {len(base_instructions)} eval clients to {len(filtered_instructions)} available clients")
            return filtered_instructions
        return base_instructions


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


