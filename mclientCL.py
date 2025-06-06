from clutils.ParamFns import set_parameters, get_parameters
from models.SimpleCNN import Net
from workloads.CIFAR10CL import load_datasets 
from clutils.make_experiences import split_dataset
from clutils.clstrat import make_cl_strat 

import json
import wandb
import os
import logging.config
import yaml
from datetime import datetime

from avalanche.benchmarks.utils import as_classification_dataset, AvalancheDataset
from avalanche.benchmarks.scenarios.dataset_scenario import benchmark_from_datasets
from avalanche.benchmarks.utils.data import make_avalanche_dataset
from avalanche.benchmarks.utils.utils import as_avalanche_dataset

import flwr
import torch
from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Metrics, Context, ConfigRecord

# Setting up Logger
# with open("config/logger.yaml", "r") as f:
#    config = yaml.safe_load(f)

# log_filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# config['handlers']['file']['filename'] = log_filename

# logging.config.dictConfig(config)

# logger = logging.getLogger("myLogger")

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
NUM_CLIENTS = 5

# Persistent State of Clients
partition_strategies = [make_cl_strat(Net().to(DEVICE)) for _ in range(NUM_CLIENTS)]


class FlowerClient(NumPyClient):
    def __init__(self, context: Context, net, benchmark, trainlen_per_exp, testlen_per_exp, partition_id, n_experiences):
        self.client_state = (context.state)
        if "local_eval_metrics" not in self.client_state.config_records:
            self.client_state.config_records["local_eval_metrics"] = ConfigRecord()
#        if "acc" not in self.client_state.config_records:
#            self.client_state.config_records["acc"] = ConfigRecord()
#        if "loss" not in self.client_state.config_records:
#            self.client_state.config_records["loss"] = ConfigRecord()
        self.net = net
        self.benchmark = benchmark
        self.trainlen_per_exp = trainlen_per_exp
        self.testlen_per_exp = testlen_per_exp
        self.cl_strategy, self.evaluation = partition_strategies[partition_id]
        self.partition_id = partition_id
        self.n_experiences = n_experiences

        print(self.client_state.config_records)

    def get_parameters(self, config):
        return get_parameters(self.cl_strategy.model)

    def fit(self, parameters, config):
        set_parameters(self.cl_strategy.model, parameters)
        rnd = config["server_round"]
        num_rounds = config["num_rounds"]

        print(f"-------------------------------------------------------------------Client {self.partition_id} Fit on round: {rnd}")
        logger.info(f"Client {self.partition_id} Fit on Round: {rnd}")

        print(dir(self.benchmark.train_stream))

        print("Starting Training")
        logger.info("Starting Training")
        results = []
        for i, experience in enumerate(self.benchmark.train_stream, start=1):
            if i == rnd:
                logger.info(f"Training on Experience : {experience.current_experience}")
                trainres = self.cl_strategy.train(experience)
                print('Training completed: ', experience.current_experience)
                logger.info(f"Training Completed: {experience.current_experience}")

        print(f"Local Evaluation of client {self.partition_id} on round {rnd}")
        logger.info(f"Local Eval of Client {self.partition_id} on round {rnd}")
        results.append(self.cl_strategy.eval(self.benchmark.test_stream))

#        print('--------------------------------RES_TRAIN----------------------------------')
#        print(res)
#        print('-------------------------------RES_STREAM_EVAL-----------------------------')
#        print(results_stream_dict)
#        print('-------------------------------GET_ALL_METRICS-----------------------------')
#        results_get_all = self.evaluation.get_all_metrics()
#        print(results_get_all)
#        print('---------------------------------LAST_METRICS------------------------------')
#        print(results_stream)

#        print('----------------------------RESULTS OF LOCAL EVAL---------------------------')
#        print(results)
        exp_acc = []
        for res in results:
            for exp, acc in res.items():
                if exp.startswith("Top1_Acc_Exp/"):
                    exp_acc.append(float(acc))
                 

        results_stream = self.evaluation.get_last_metrics()
        confusion_matrix = results_stream["ConfusionMatrix_Stream/eval_phase/test_stream"].tolist()
#        forgetting_measure = results_stream["StreamForgetting/eval_phase/test_stream"]
        stream_loss = results_stream["Loss_Stream/eval_phase/test_stream"]
        stream_acc = results_stream["Top1_Acc_Stream/eval_phase/test_stream"]
        stream_disc_usage = results_stream["DiskUsage_Stream/eval_phase/test_stream"]

        local_eval_metrics = self.client_state.config_records["local_eval_metrics"]
        forgetting_per_exp = []
        if rnd  == num_rounds:
            print(f"----------------------------Calculating Forgetting Measures for Client {self.partition_id} in round: {rnd}------------------------------")
            exp_acc_hist = local_eval_metrics["expacc"]
            print(exp_acc_hist)
            print(exp_acc)
            for i, e in enumerate(exp_acc_hist):
                e = json.loads(e)
                print(f"LOCAL FORGETTING for EXP{i+1} in client {self.partition_id} = Initial - Final")
                print(f"{e[i]} - {exp_acc[i]} = {e[i] - exp_acc[i]}")
                fm = e[i] - exp_acc[i];
                forgetting_per_exp.append(fm)
            print("Forgetting per exp", forgetting_per_exp)
            forgetting_measure = sum(forgetting_per_exp)/self.n_experiences
        else:
            forgetting_measure = 0
                


        fit_dict_return = {
                "confusion_matrix": json.dumps(confusion_matrix),
                "forgetting_measure":  float(forgetting_measure),
                "stream_loss":  float(stream_loss),
                "stream_acc":  float(stream_acc),
                "stream_disc_usage":  float(stream_disc_usage),
                "exp_acc": json.dumps(exp_acc),
                "forgetting_per_exp": json.dumps(forgetting_per_exp),
                "pid": self.partition_id,
                "round": rnd,
            }
        print("----------------------------CLIENT_INFO--------------------------------")
        print(fit_dict_return)
        print('-----------------------------------------------------------------------')

        
# Logging State
        if rnd != 0:
            if "expacc" not in local_eval_metrics:
                local_eval_metrics["expacc"] = [json.dumps(exp_acc)]
            else:
                local_eval_metrics["expacc"].append(json.dumps(exp_acc))
            if "acc" not in local_eval_metrics:
                local_eval_metrics["acc"] = [stream_acc]
            else:
                local_eval_metrics["acc"].append(stream_acc)
            if "loss" not in local_eval_metrics:
                local_eval_metrics["loss"] = [stream_loss]
            else:
                local_eval_metrics["loss"].append(stream_loss)
        if rnd == num_rounds:
            local_eval_metrics["forgetting_per_exp"] = [json.dumps(forgetting_per_exp)]
            local_eval_metrics["forgetting"] = [forgetting_measure]
            

        
        return get_parameters(self.cl_strategy.model), self.trainlen_per_exp[rnd-1], fit_dict_return

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)

        rnd = config["server_round"]
        num_rounds = config["num_rounds"]

        cl_strategy, evaluation = make_cl_strat(self.net)
        results = []
        print("------------------------Evaluating Client for Server on Updated Global Model on Test Set--------------------")
        results.append(cl_strategy.eval(self.benchmark.test_stream))
##        loss, accuracy = test(self.net, self.valloader)
        last_metrics = evaluation.get_last_metrics()
        loss = last_metrics["Loss_Stream/eval_phase/test_stream"]
        accuracy = last_metrics["Top1_Acc_Stream/eval_phase/test_stream"]
        print("Results of Eval for GCF----------------------------------------------------------------")
        print(results)
        exp_acc = []
        for res in results:
            for exp, acc in res.items():
                if exp.startswith("Top1_Acc_Exp/"):
                    exp_acc.append(float(acc))
                    


        print("Eval of Client: ")
        print("Loss: ", loss)
        print("Acc: ", accuracy)
        print("Per Exp Acc: ", exp_acc)

        eval_dict_return = {
                "accuracy": float(accuracy),
                "loss": float(loss),
                "ExpAccuracy": json.dumps(exp_acc),
                "server_round": rnd,
                "pid": self.partition_id,
                }

        return float(loss), sum(self.testlen_per_exp), eval_dict_return


def client_fn(context: Context) -> Client:
    """Create a Flower client representing a single organization."""

    # Load model
    net = Net().to(DEVICE)

    # Load data (CIFAR-10)
    # Note: each client gets a different trainloader/valloader, so each client
    # will train and evaluate on their own unique data partition
    # Read the node_config to fetch data partition associated to this node
    partition_id = context.node_config["partition-id"]
    train_data, test_data = load_datasets(partition_id=partition_id)

    total_train_samples = len(train_data)
    total_eval_samples = len(test_data)
    trainloader_len = total_train_samples//BATCH_SIZE
    testloader_len = total_eval_samples//BATCH_SIZE

    print(f"Context: {context}")

    # Splitting Data into Experiences

    n_experiences = 5
    train_experiences = split_dataset(train_data, n_experiences)
    test_experiences = split_dataset(test_data, n_experiences)  # optional

    ava_train = []
    ava_test = []

    trainlen_per_exp = []
#    ava_test.append(as_avalanche_dataset(test_data))
    for i, exp in enumerate(train_experiences, start=1):
#        print(f"Len of  Train Exp {i}: {len(exp)}")
        trainlen_per_exp.append(len(exp))
        ava_exp = as_avalanche_dataset(exp)
#        print(type(ava_exp))
        ava_train.append(exp)

#    Provision for Splitting Test Stream 

    testlen_per_exp = []
    for i, exp in enumerate(test_experiences):
#        print(f"Len of Test Exp {i}: {len(exp)}")
        testlen_per_exp.append(len(exp))
        ava_exp = as_avalanche_dataset(exp)
        ava_test.append(exp)

    # Creating Bencmarks -> using Entire testdata for eval -> have to check difference

    benchmark = benchmark_from_datasets(train=ava_train, test=ava_test)

    # Print ClientID

    print(f"---------------------------------LAUNCHING CLIENT: {partition_id}--------------------------------------------------")

    # Create a single Flower client representing a single organization
    # FlowerClient is a subclass of NumPyClient, so we need to call .to_client()
    # to convert it to a subclass of `flwr.client.Client`
    return FlowerClient(context, net, benchmark, trainlen_per_exp, testlen_per_exp, partition_id, n_experiences).to_client()


