from typing import List, Tuple
from flwr.common import Metrics
from logging import INFO, WARNING
from flwr.common.logger import log

import wandb
import  os

import numpy as np
import json

NUM_ROUNDS = 5

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
#        id = f"{run_id}"
        )

# State of all rounds metrics
wexpacc_byround = []
gcf_per_exp_running = [0] * NUM_ROUNDS 


def evaluate_metrics_aggregation_fn(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    losses = [num_examples * m["loss"] for num_examples, m in metrics]
    pid = [m["pid"] for _, m in metrics]
    rnd = metrics[0][1]["server_round"]

    exp_accuracy_ds = [json.loads(m["ExpAccuracy"]) for _, m in metrics]

#    exp_accuracy = [m["ExpAccuracy"] for _, m in metrics]
    client_accuracy = [m["accuracy"] for _, m in metrics]
    client_loss = [m["loss"] for _, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    
    wexpacc = [sum(w * val for w, val in zip(examples, values))/sum(examples) for values in zip(*exp_accuracy_ds)]

    wexpacc_byround.append(wexpacc)

    latest_wexpacc = wexpacc_byround[-1]

    for i, w in enumerate(wexpacc_byround):
        gcf_per_exp_running[i] = wexpacc_byround[i][i] - latest_wepacc[i]

    gcf = sum(gcf_per_exp_running)/NUM_ROUNDS

    wandb.log(
            {
            "global_accuracy": sum(accuracies) / sum(examples),
            "global_loss": sum(losses) / sum(examples),
            "gcf": gcf,
            "round": rnd,
            }, step=rnd
            )
    for i, expacc in enumerate(wexpacc, start=1):
        wandb.log({
            f"global_exp_{i}_accuracy": expacc,
            "round": rnd,
            }, step=rnd
            )
    for i, (acc, cid) in enumerate(zip(client_accuracy, pid), start=1):
        wandb.log({
            f"global_acc_client_{cid}": acc,
            "round": rnd,
            }, step = rnd
            )
    for i, (ls, cid) in enumerate(zip(client_loss, pid), start=1):
        wandb.log({
            f"global_loss_client_{cid}": ls,
            "round": rnd,
            }, step = rnd
            )
    for i, gcf in enumerate(gcf_per_exp_running):
        wandb.log({
            f"gcf_exp_{i}": gcf,
            "round": rnd,
            }, step = rnd
            )


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
    local_fm = [m["forgetting_measure"] for _, m in metrics]

    rnd = metrics[0][1]["round"]
    pid = [m["pid"] for _, m in metrics]

    exp_accuracy_ds = [json.loads(m["exp_acc"]) for _, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    wexpacc = [sum(w * val for w, val in zip(examples, values))/sum(examples) for values in zip(*exp_accuracy_ds)]

    wandb.log(
            {
            "local_eval_accuracy": sum(accuracies) / sum(examples),
            "local_eval_loss": sum(losses) / sum(examples),
            "round": rnd
            }, step = rnd
            )

    for (cid, fm) in zip(pid, local_fm):
        wandb.log({
            f"local_fm_client_{cid}": fm,
            "round": rnd,
            }, step = rnd
            )

                


    return {
            "local_eval_accuracy": sum(accuracies) / sum(examples),
            "local_eval_loss": sum(losses) / sum(examples),
            "wexp_accuracy": ",".join(map(str, wexpacc)),
            "local_fm": ",".join(map(str, local_fm)),
            }



    
