import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from avalanche.evaluation.metrics import (
    forgetting_metrics, 
    accuracy_metrics,
    loss_metrics,
    timing_metrics,
    cpu_usage_metrics, 
    disk_usage_metrics,
    )
from avalanche.logging import InteractiveLogger, TextLogger, TensorboardLogger
from avalanche.training.plugins import EvaluationPlugin

from avalanche.training.supervised import Naive

from omegaconf import OmegaConf
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config_utils import load_config

cfg = load_config()

stratetgy = cfg.cl.strategy 
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def make_cl_strat(net):
    # log to text file
    # text_logger = TextLogger(open(f"{cfg.}"))
    text_logger = TextLogger(open('logs/avalog2.txt', 'a'))

    # print to stdout
    interactive_logger = InteractiveLogger()

    # Only compute standard metrics (no confusion matrix)
    metrics = [
        accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        timing_metrics(epoch=True, epoch_running=True),
        forgetting_metrics(experience=True, stream=True),
        cpu_usage_metrics(experience=True),
        disk_usage_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    ]

    eval_plugin = EvaluationPlugin(
        *metrics,
        loggers=[interactive_logger, text_logger]
    )

    cl_strategy = Naive(
        model=net, 
        optimizer=Adam(net.parameters(), lr=cfg.training.learning_rate),
        criterion=CrossEntropyLoss(), 
        train_mb_size=cfg.dataset.batch_size, 
        train_epochs=cfg.client.epochs, 
        eval_mb_size=cfg.dataset.batch_size,
        evaluator=eval_plugin,
        device=DEVICE
    )
    return cl_strategy, eval_plugin
