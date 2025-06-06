import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from avalanche.evaluation.metrics import (
    forgetting_metrics, 
    accuracy_metrics,
    loss_metrics,
    timing_metrics,
    cpu_usage_metrics, 
    confusion_matrix_metrics,
    disk_usage_metrics,
    )
from avalanche.logging import InteractiveLogger, TextLogger, TensorboardLogger, WandBLogger
from avalanche.training.plugins import EvaluationPlugin

from avalanche.training.supervised import Naive

from omegaconf import OmegaConf
from pathlib import Path

config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'
cfg = OmegaConf.load(config_path)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def make_cl_strat(net):

    # log to text file
    text_logger = TextLogger(open('logs/avalog.txt', 'a'))

    # print to stdout
    interactive_logger = InteractiveLogger()

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        timing_metrics(epoch=True, epoch_running=True),
        forgetting_metrics(experience=True, stream=True),
        cpu_usage_metrics(experience=True),
        confusion_matrix_metrics(num_classes=10, save_image=False,
                                 stream=True),
        disk_usage_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loggers=[interactive_logger, text_logger]
    )

    cl_strategy = Naive(
        net, Adam(net.parameters()),
        CrossEntropyLoss(), train_mb_size=32, train_epochs=3, eval_mb_size=32,
        evaluator=eval_plugin,
        device=DEVICE
        )
    return cl_strategy, eval_plugin
