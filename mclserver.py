import time

import flwr
from flwr.common import Context
from flwr.server import ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

import wandb

from clutils.scallbacks import (
    evaluate_metrics_aggregation_fn,
    fit_metrics_aggregation_fn,
    fit_config,
    eval_config,
)
from config_utils import load_config
from utils.latency_simulator import get_runtime_recorder

cfg = load_config()

NUM_ROUNDS = cfg.server.num_rounds
NUM_CLIENTS = cfg.server.num_clients

class LatencyAwareFedAvg(FedAvg):
    def aggregate_fit(self, server_round, results, failures):  # noqa: D401
        dropped_clients = []
        dropped_entries = []
        filtered_results = []
        for client_proxy, fit_res in results:
            metrics = dict(fit_res.metrics)
            if metrics.get("latency/dropped", False):
                client_id = int(metrics.get("pid", -1))
                dropped_clients.append(client_id)
                dropped_entries.append((fit_res.num_examples, metrics))
                continue
            filtered_results.append((client_proxy, fit_res))

        start_time = time.time()
        aggregated = super().aggregate_fit(server_round, filtered_results, failures)
        duration = time.time() - start_time

        recorder = get_runtime_recorder()
        if recorder is not None:
            recorder.log_server_round(
                round_id=server_round,
                aggregation_time_s=duration,
                total_results=len(results),
                accepted_results=len(filtered_results),
                dropped_clients=dropped_clients,
            )
            if dropped_entries:
                recorder.log_client_round(server_round, dropped_entries, dropped_clients=dropped_clients)

        wandb.log(
            {
                "server/aggregation_time_s": duration,
                "server/total_results": len(results),
                "server/accepted_results": len(filtered_results),
                "server/dropped_clients": len(dropped_clients),
            },
            step=server_round,
        )
        return aggregated


# Create FedAvg strategy
strategy = LatencyAwareFedAvg(
    fraction_fit=cfg.server.fraction_fit,
    fraction_evaluate=cfg.server.fraction_eval,
    min_fit_clients=cfg.server.min_fit,
    min_evaluate_clients=cfg.server.min_eval,
    min_available_clients=cfg.server.num_clients,
    on_fit_config_fn=fit_config,
    on_evaluate_config_fn=eval_config,
    evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
    fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
)

def server_fn(context: Context) -> ServerAppComponents:
    """Construct components that set the ServerApp behaviour.

    You can use the settings in `context.run_config` to parameterize the
    construction of all elements (e.g the strategy or the number of rounds)
    wrapped in the returned ServerAppComponents object.
    """

    # Configure the server for 5 rounds of training
    config = ServerConfig(cfg.server.num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


