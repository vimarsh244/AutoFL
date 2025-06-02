from typing import List, Tuple
from flwr.common import Metrics
from logging import INFO, WARNING
from flwr.common.logger import log

import wandb
import  os

import numpy as np
import json

# WandB Initialization
run_id = os.getenv("RUN_ID")
wandb.init(
        project = "test-autofl",
        config={
            "dataset":  "cifar10",
            "num_clients":  5,
            "num_rounds":  5,
            "local_epochs":  3,
            },
        id = f"{run_id}"
        )

def evaluate_metrics_aggregation_fn(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    losses = [num_examples * m["loss"] for num_examples, m in metrics]

    exp_accuracy_ds = [json.loads(m["ExpAccuracy"]) for _, m in metrics]

#    exp_accuracy = [m["ExpAccuracy"] for _, m in metrics]
    client_accuracy = [m["accuracy"] for _, m in metrics]
    client_loss = [m["loss"] for _, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    
    wexpacc = [sum(w * val for w, val in zip(examples, values))/sum(examples) for values in zip(*exp_accuracy_ds)]

    wandb.log(
            {
            "global_accuracy": sum(accuracies) / sum(examples),
            "global_loss": sum(losses) / sum(examples),
            "global_wexp_accuracy": ",".join(map(str, wexpacc)),
            "global_client_accuracy": ",".join(map(str, client_accuracy)),
            "global_client_loss": ",".join(map(str, client_loss)),
            "global_exp_accuracy": json.dumps(exp_accuracy_ds),
            }
            )
                


    return {
            "accuracy": sum(accuracies) / sum(examples),
            "loss": sum(losses) / sum(examples),
            "wexp_accuracy": ",".join(map(str, wexpacc)),
            "client_accuracy": ",".join(map(str, client_accuracy)),
            "client_loss": ",".join(map(str, client_loss)),
            "exp_accuracy": json.dumps(exp_accuracy_ds),
            }

def fit_metrics_aggregation_fn(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["stream_acc"] for num_examples, m in metrics]
    losses = [num_examples * m["stream_loss"] for num_examples, m in metrics]
    disc_usage = [m["stream_disc_usage"] for _, m in metrics]

    exp_accuracy_ds = [json.loads(m["exp_acc"]) for _, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    wexpacc = [sum(w * val for w, val in zip(examples, values))/sum(examples) for values in zip(*exp_accuracy_ds)]

    wandb.log(
            {
            "local_train_eval_accuracy": sum(accuracies) / sum(examples),
            "local_train_eval_loss": sum(losses) / sum(examples),
            "local_train_eval_wexp_accuracy": ",".join(map(str, wexpacc)),
            }
            )
                


    return {
            "train_eval_accuracy": sum(accuracies) / sum(examples),
            "train_eval_loss": sum(losses) / sum(examples),
            "wexp_accuracy": ",".join(map(str, wexpacc)),
            }



    
