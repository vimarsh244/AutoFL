from typing import List, Tuple
from flwr.common import Metrics
from logging import INFO, WARNING
from flwr.common.logger import log

import numpy as np
import json

def evaluate_metrics_aggregation_fn(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    losses = [num_examples * m["loss"] for num_examples, m in metrics]

    exp_accuracy_ds = [json.loads(m["ExpAccuracy"]) for _, m in metrics]

#    exp_accuracy = [m["ExpAccuracy"] for _, m in metrics]
    client_accuracy = [m["accuracy"] for _, m in metrics]
    client_loss = [m["loss"] for _, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    
    wexpacc = [sum(w * val for w, val in zip(examples, values))/sum(examples) for values in zip(*exp_accuracy_ds)]


    return {
            "accuracy": sum(accuracies) / sum(examples),
            "loss": sum(losses) / sum(examples),
            "wexp_accuracy": ",".join(map(str, wexpacc)),
            "client_accuracy": ",".join(map(str, client_accuracy)),
            "client_loss": ",".join(map(str, client_loss)),
            "exp_accuracy": json.dumps(exp_accuracy_ds),
            }

# def fit_metrics_aggregation_fn(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    
