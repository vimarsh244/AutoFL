"""Latency-aware Flower server strategies."""

from __future__ import annotations

import time
from typing import List, Sequence, Tuple

from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy
from flwr.common import FitRes

import wandb

from utils.latency_simulator import get_runtime_recorder


class LatencyAwareFedAvg(FedAvg):
    """FedAvg variant that skips dropped clients and logs aggregation runtime."""

    def aggregate_fit(
        self,
        server_round: int,
        results: Sequence[Tuple[ClientProxy, FitRes]],
        failures: Sequence[BaseException],
    ):
        dropped_clients: List[int] = []
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
                recorder.log_client_round(
                    server_round,
                    dropped_entries,
                    dropped_clients=dropped_clients,
                )

        wandb.log(
            {
                "server/aggregation_time_s": duration,
                "server/total_results": len(results), # KIND OF Useless (but keeping)
                "server/accepted_results": len(filtered_results),
                "server/dropped_clients": len(dropped_clients),
            },
            step=server_round,
        )
        return aggregated


STRATEGY_REGISTRY = {
    "latency_aware_fedavg": LatencyAwareFedAvg,
    "latencyawarefedavg": LatencyAwareFedAvg,
    "fedavg": LatencyAwareFedAvg,
}
