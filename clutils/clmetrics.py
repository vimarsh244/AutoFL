from typing import List, Tuple
from flwr.common import Metrics
from logging import INFO, WARNING
from flwr.common.logger import log
import numpy as np

def evaluate_metrics_aggregation_fn(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    losses = [num_examples * m["loss"] for num_examples, m in metrics]
    client_accuracy = [m["accuracy"] for _, m in metrics]
    client_loss = [m["loss"] for _, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {
            "accuracy": sum(accuracies) / sum(examples),
            "loss": sum(losses) / sum(examples),
            "client_accuracy": ",".join(map(str, client_accuracy)),
            "client_loss": ",".join(map(str, client_loss)),
            }

# def fit_metrics_aggregation_fn(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    
