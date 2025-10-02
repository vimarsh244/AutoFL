from typing import List, Tuple
from flwr.common import Metrics
from logging import INFO, WARNING
from flwr.common.logger import log

import wandb
import os
from omegaconf import OmegaConf
from pathlib import Path

import numpy as np
import json

# Setup Config
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config_utils import load_config
cfg = load_config()

# Only initialize wandb if mode is not disabled
if cfg.wb.get('mode', 'online') != 'disabled':
    wandb.init(
        project=cfg.wb.project,
        name=cfg.wb.name,
        config=OmegaConf.to_container(cfg, resolve=True),
        mode=cfg.wb.get('mode', 'online')
    )
else:
    # Create a dummy run for disabled mode
    class DummyWandB:
        def log(self, *args, **kwargs):
            pass
        class plot:
            @staticmethod
            def confusion_matrix(*args, **kwargs):
                return None
    wandb = DummyWandB()

NUM_ROUNDS = cfg.server.num_rounds
LOCAL_EPOCHS = cfg.client.epochs
NUM_CLIENTS = cfg.server.num_clients

# State of all rounds metrics

def fit_config(server_round: int):
    """Return training configuration dict for each round."""
    config = {
        "server_round": server_round,  
        "local_epochs": cfg.client.epochs, 
        "num_rounds": cfg.server.num_rounds
    }
    return config

def eval_config(server_round: int):
    config = {
            "server_round": server_round,
            "local_epochs": cfg.client.epochs,
            "num_rounds": cfg.server.num_rounds,
            }
    return config

def evaluate_metrics_aggregation_fn(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    client_accuracies = [m["stream_accuracy"] for _, m in metrics]
    client_losses = [m["stream_loss"] for _, m in metrics]
    w_accuracies = [num_examples * m["stream_accuracy"] for num_examples, m in metrics]
    w_losses = [num_examples * m["stream_loss"] for num_examples, m in metrics]
    pid = [m["pid"] for _, m in metrics]
    rnd = metrics[0][1]["server_round"]

    cumalative_forgetting_measures = [m["cumalative_forgetting_measure"] for _, m in metrics]
    stepwise_forgetting_measures = [m["stepwise_forgetting_measure"] for _, m in metrics]

    accuracypexp_pc = [json.loads(m["accuracy_per_experience"]) for _, m in metrics]

    examples = [num_examples for num_examples, _ in metrics]
    weighted_accuracy_pexp = [sum(w * val for w, val in zip(examples, values))/sum(examples) for values in zip(*accuracypexp_pc)]

    eval_metrics = {
        "global/average/accuracy": sum(w_accuracies) / sum(examples),
        "global/client/accuracy": {id: acc for id, acc in zip(pid,client_accuracies)},
        "global/average/loss": sum(w_losses) / sum(examples),
        "global/client/loss": {id: loss for id, loss in zip(pid, client_losses)},
        "global/average/cumalative_forgetting": sum(cumalative_forgetting_measures) / len(cumalative_forgetting_measures),
        "global/client/cumalative_forgetting": {id: cmfm for id, cmfm in zip(pid, cumalative_forgetting_measures)},
        "global/average/stepwise_forgetting": sum(stepwise_forgetting_measures) /  len(stepwise_forgetting_measures),
        "global/client/stepwise_forgetting": {id: swfm for id, swfm in zip(pid, stepwise_forgetting_measures)},
        "global/experience/accuracy": {id: acc for id, acc in zip(pid, weighted_accuracy_pexp)},
    }


    wandb.log(eval_metrics, step=rnd)

    return {
            "global/average_accuracy": sum(w_accuracies) / sum(examples),
            "global/average_loss": sum(w_losses) / sum(examples),
            "global/average_cumalative_forgetting": sum(cumalative_forgetting_measures) / len(cumalative_forgetting_measures),
            "global/average_stepwise_forgetting": sum(stepwise_forgetting_measures) / len(stepwise_forgetting_measures),
            }

def fit_metrics_aggregation_fn(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Calculate Metrics After Fit of Clients"""

    # Per Client Acc and Loss
    client_acc = [m["stream_acc"] for _, m in metrics]
    client_loss = [m["stream_loss"] for _, m in metrics]
    # Weighted Acc and Loss
    w_accuracies = [num_examples * m["stream_acc"] for num_examples, m in metrics]
    w_losses = [num_examples * m["stream_loss"] for num_examples, m in metrics]
    # Forgetting Measures
    cumalative_forgetting_measures = [m["cumalative_forgetting_measure"] for _, m in metrics]
    stepwise_forgetting_measures = [m["stepwise_forgetting_measure"] for _, m in metrics]

    # Round and Partition Id's
    rnd = metrics[0][1]["round"]
    pid = [m["pid"] for _, m in metrics]

    accuracy_per_exp_pc = [json.loads(m["accuracy_per_experience"]) for _, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    weighted_accuracy_per_exp = [sum(w * val for w, val in zip(examples, values))/sum(examples) for values in zip(*accuracy_per_exp_pc)]

    fit_metrics = {
        "local/average/accuracy": sum(w_accuracies) / sum(examples),
        "local/client/accuracy": {id: acc for id, acc in zip(pid,client_acc)},
        "local/average/loss": sum(w_losses) / sum(examples),
        "local/client/loss": {id: loss for id, loss in zip(pid, client_loss)},
        "local/average/cumalative_forgetting": sum(cumalative_forgetting_measures) / len(cumalative_forgetting_measures),
        "local/client/cumalative_forgetting": {id: cmfm for id, cmfm in zip(pid, cumalative_forgetting_measures)},
        "local/average/stepwise_forgetting": sum(stepwise_forgetting_measures) / len(stepwise_forgetting_measures),
        "local/client/stepwise_forgetting": {id: swfm for id, swfm in zip(pid, stepwise_forgetting_measures)},
        "local/experience/accuracy": {id: acc for id, acc in zip(pid, weighted_accuracy_per_exp)},
        }

    # Logging to Wandb
    wandb.log(fit_metrics, step=rnd)

    return {
            "local/average_accuracy": sum(w_accuracies) / sum(examples),
            "local/average_loss": sum(w_losses) / sum(examples),
            "local/average_cumalative_forgetting": sum(cumalative_forgetting_measures)/ len(cumalative_forgetting_measures),
            "local/average_stepwise_forgetting": sum(stepwise_forgetting_measures) /  len(stepwise_forgetting_measures),
            }   
