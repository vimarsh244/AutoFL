from utils.ParamFns import set_parameters, get_parameters
from utils.TrainTestFns import train, test
from models.SimpleCNN import Net
from workloads.CIFAR10CL import load_datasets 
from clutils.make_experiences import split_dataset
from clutils.clstrat import make_cl_strat 

from avalanche.benchmarks.utils import as_classification_dataset, AvalancheDataset
from avalanche.benchmarks.scenarios.dataset_scenario import benchmark_from_datasets
from avalanche.benchmarks.utils.data import make_avalanche_dataset
from avalanche.benchmarks.utils.utils import as_avalanche_dataset

import flwr
import torch
from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Metrics, Context

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
NUM_CLIENTS = 5

class FlowerClient(NumPyClient):
    def __init__(self, net, benchmark, trainloader_len, testloader_len):
        self.net = net
        self.benchmark = benchmark
        self.trainloader_len = trainloader_len
        self.testloader_len = testloader_len

    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
            # TRAINING LOOP
        print('Starting experiment...')
        for experience in self.benchmark.train_stream:
            print("Start of experience: ", experience.current_experience)

            # train returns a dictionary which contains all the metric values

            cl_strategy, evaluation = make_cl_strat(self.net)
            res = cl_strategy.train(experience)
            print('Training completed: ', experience.current_experience)
           ## train(self.net, self.trainloader, epochs=1)
        return get_parameters(self.net), self.trainloader_len, {}

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        cl_strategy, evaluation = make_cl_strat(self.net)
        results = []
        for experience in self.benchmark.train_stream:
            print("Computing acc on whole test set")
            results.append(cl_strategy.eval(self.benchmark.test_stream))

            
##        loss, accuracy = test(self.net, self.valloader)
        last_metrics = evaluation.get_last_metrics()
        loss = last_metrics["Loss_Stream/eval_phase/test_stream"]
        accuracy = last_metrics["Top1_Acc_Stream/eval_phase/test_stream"]

        print("Eval of Client: ")
        print("Loss: ", loss)
        print("Acc: ", accuracy)

        return float(loss), self.testloader_len, {"accuracy": float(accuracy)}


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
    print(type(train_data))
    total_train_samples = len(train_data)
    total_eval_samples = len(test_data)
    trainloader_len = total_train_samples//BATCH_SIZE
    testloader_len = total_eval_samples//BATCH_SIZE

    # Splitting Data into Experiences

    n_experiences = 5
    train_experiences = split_dataset(train_data, n_experiences)
    test_experiences = split_dataset(test_data, n_experiences)  # optional
    ava_train = []
    ava_test = []
    ava_test.append(as_avalanche_dataset(test_data))
    for exp in train_experiences:
        ava_exp = as_avalanche_dataset(exp)
        print(type(ava_exp))
        ava_train.append(exp)

    # Creating Bencmarks -> using Entire testdata for eval -> have to check difference

    print(type(ava_train))
    benchmark = benchmark_from_datasets(train=ava_train, test=ava_test)

    # Print ClientID

    print("ClientID: ", context.node_config["partition-id"])


    # Create a single Flower client representing a single organization
    # FlowerClient is a subclass of NumPyClient, so we need to call .to_client()
    # to convert it to a subclass of `flwr.client.Client`
    return FlowerClient(net, benchmark, trainloader_len, testloader_len).to_client()


