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
config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'
cfg = OmegaConf.load(config_path)

NUM_ROUNDS = cfg.server.num_rounds
LOCAL_EPOCHS = cfg.client.epochs
NUM_CLIENTS = cfg.server.num_clients

# State of all rounds metrics
wexpacc_byround = []
gcf_per_exp_running =[0 for _ in range (NUM_ROUNDS)]

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
            "local_epochs": 3,
            "num_rounds": NUM_ROUNDS
            }
    return config

def evaluate_metrics_aggregation_fn(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    losses = [num_examples * m["loss"] for num_examples, m in metrics]
    pid = [m["pid"] for _, m in metrics]
    rnd = metrics[0][1]["server_round"]

    exp_accuracy_ds = [json.loads(m["ExpAccuracy"]) for _, m in metrics]
    client_accuracy = [m["accuracy"] for _, m in metrics]
    client_loss = [m["loss"] for _, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    
    wexpacc = [sum(w * val for w, val in zip(examples, values))/sum(examples) for values in zip(*exp_accuracy_ds)]
    wexpacc_byround.append(wexpacc)
    latest_wexpacc = wexpacc_byround[-1]

    for i, w in enumerate(wexpacc_byround):
        gcf_per_exp_running[i] = wexpacc_byround[i][i] - latest_wexpacc[i]

    gcf = sum(gcf_per_exp_running)/NUM_ROUNDS

    # Log all metrics in a single wandb.log call for better organization
    wandb_metrics = {
        # Global metrics
        "global/accuracy": sum(accuracies) / sum(examples),
        "global/loss": sum(losses) / sum(examples),
        "global/forgetting_measure": gcf,
        "round": rnd,
    }

    # Per-experience metrics
    for i, expacc in enumerate(wexpacc, start=1):
        wandb_metrics[f"global/experience_{i}/accuracy"] = expacc

    # Per-client metrics
    for i, (acc, cid) in enumerate(zip(client_accuracy, pid), start=1):
        wandb_metrics[f"client/{cid}/accuracy"] = acc
        wandb_metrics[f"client/{cid}/loss"] = client_loss[i-1]

    # Per-experience forgetting metrics
    for i, g in enumerate(gcf_per_exp_running):
        wandb_metrics[f"global/experience_{i}/forgetting"] = g

    # Log all metrics at once
    wandb.log(wandb_metrics, step=rnd)

    return {
        "global_accuracy": sum(accuracies) / sum(examples),
        "global_loss": sum(losses) / sum(examples),
        "global_wexp_accuracy": ",".join(map(str, wexpacc)),
        "global_client_accuracy": ",".join(map(str, client_accuracy)),
        "global_client_loss": ",".join(map(str, client_loss)),
        "global_exp_accuracy": json.dumps(exp_accuracy_ds),
        "gcf": float(gcf),
        "gcf_per_exp": json.dumps(gcf_per_exp_running),
    }

def fit_metrics_aggregation_fn(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["stream_acc"] for num_examples, m in metrics]
    losses = [num_examples * m["stream_loss"] for num_examples, m in metrics]
    disc_usage = [m["stream_disc_usage"] for _, m in metrics]
    local_fm = [m["cumalative_forgetting_measure"] for _, m in metrics]

    rnd = metrics[0][1]["round"]
    pid = [m["pid"] for _, m in metrics]

    exp_accuracy_ds = [json.loads(m["accuracy_per_experience"]) for _, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    wexpacc = [sum(w * val for w, val in zip(examples, values))/sum(examples) for values in zip(*exp_accuracy_ds)]

    # Log all metrics in a single wandb.log call for better organization
    wandb_metrics = {
        # Local evaluation metrics
        "local/accuracy": sum(accuracies) / sum(examples),
        "local/loss": sum(losses) / sum(examples),
        "round": rnd,
    }

    # Per-client forgetting metrics
    for cid, fm in zip(pid, local_fm):
        wandb_metrics[f"client/{cid}/forgetting_measure"] = fm

    # Per-experience accuracy
    for i, acc in enumerate(wexpacc, start=1):
        wandb_metrics[f"local/experience_{i}/accuracy"] = acc

    # Log all metrics at once
    wandb.log(wandb_metrics, step=rnd)

    return {
        "local_eval_accuracy": sum(accuracies) / sum(examples),
        "local_eval_loss": sum(losses) / sum(examples),
        "wexp_accuracy": ",".join(map(str, wexpacc)),
        "local_fm": ",".join(map(str, local_fm)),
    }



    
