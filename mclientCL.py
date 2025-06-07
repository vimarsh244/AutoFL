# Import From Custom Modules
from clutils.ParamFns import set_parameters, get_parameters
from clutils.make_experiences import split_dataset
from clutils.clstrat import make_cl_strat 

#Import basic Modules
import json
import random
import os
import warnings
from omegaconf import OmegaConf
# Avalanche Imports
from avalanche.benchmarks.utils import as_classification_dataset, AvalancheDataset
from avalanche.benchmarks.scenarios.dataset_scenario import benchmark_from_datasets
from avalanche.benchmarks.utils.data import make_avalanche_dataset
from avalanche.benchmarks.utils.utils import as_avalanche_dataset

# Flower Imports
import flwr
import torch
from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Metrics, Context, ConfigRecord

# Ignore Flower Warnings
warnings.filterwarnings("ignore")

#Setting up Configuration
cfg = OmegaConf.load('config/config.yaml')


# Import workload based on configuration
if cfg.dataset.workload == "cifar10":
    from workloads.CIFAR10CL import load_datasets
elif cfg.dataset.workload == "cifar100":
    if cfg.cl.strategy == "domain":
        from workloads.CIFAR100DomainCL import load_datasets
    else:
        from workloads.CIFAR100CL import load_datasets
elif cfg.dataset.workload == "bdd100k":
    from workloads.BDD100KDomainCL import load_datasets
elif cfg.dataset.workload == "kitti":
    from workloads.KITTIDomainCL import load_datasets
else:
    raise ValueError(f"Unknown workload: {cfg.dataset.workload}")

# Setting Global Variables
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = cfg.dataset.batch_size
NUM_CLIENTS = cfg.server.num_clients
NUM_EXP = cfg.cl.num_experiences

# Enable Green Print
def cprint(text, color="green"):
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "white": "\033[97m",
        "reset": "\033[0m",
    }
    print(colors.get(color, colors["reset"]) + text + colors["reset"])

def get_model():
    """Get model based on configuration"""
    if cfg.model.name == "resnet":
        from models.ResNet import ResNet
        return ResNet(num_classes=cfg.model.num_classes)
    elif cfg.model.name == "simple_cnn":
        from models.SimpleCNN import Net
        return Net()
    else:
        raise ValueError(f"Unknown model: {cfg.model.name}")

# Persistent State of Clients
partition_strategies = [make_cl_strat(get_model().to(DEVICE)) for _ in range(NUM_CLIENTS)]

# Client Class
class FlowerClient(NumPyClient):
    def __init__(self, context: Context, net, benchmark, trainlen_per_exp, testlen_per_exp, partition_id):
        self.client_state = context.state
        if not hasattr(self.client_state, 'config_records'):
            self.client_state.config_records = ConfigRecord()
        if "local_eval_metrics" not in self.client_state.config_records:
            self.client_state.config_records["local_eval_metrics"] = ConfigRecord()
        if "availability" not in self.client_state.config_records:
            self.client_state.config_records["availability"] = ConfigRecord()
        # Special Provision for acc per exp as needed to calculate fm
        if "accuracy_per_exp" not in self.client_state.config_records["local_eval_metrics"]:
            self.client_state.config_records["local_eval_metrics"]["accuracy_per_exp"] = []
        self.net = net
        self.benchmark = benchmark
        self.trainlen_per_exp = trainlen_per_exp
        self.testlen_per_exp = testlen_per_exp
        self.cl_strategy, self.evaluation = partition_strategies[partition_id]
        self.partition_id = partition_id

        # To add  later: Battery, Location, Speed, Mobility_Trace

        print(self.client_state.config_records)

    # Get Params from Global Model
    def get_parameters(self, config):
        return get_parameters(self.cl_strategy.model)

    # Fit on Local Data
    def fit(self, parameters, config):
        set_parameters(self.cl_strategy.model, parameters)
        rnd = config["server_round"]
        num_rounds = config["num_rounds"]

        cprint("FIT")
        print(f"Client {self.partition_id} Fit on round: {rnd}")

        # Train on Experience as per Round
        cprint("Starting Training")
        results = []
        for i, experience in enumerate(self.benchmark.train_stream, start=1):
            if i == rnd:
                print(f"EXP: {experience.current_experience}")
                trainres = self.cl_strategy.train(experience)
                cprint('Training completed: ')

        # Local Eval after fit on client for metrics
        print(f"Local Evaluation of client {self.partition_id} on round {rnd}")
        results.append(self.cl_strategy.eval(self.benchmark.test_stream))

        # Calc Accuracy per Experience 
        curr_accpexp = []
        for res in results:
            for exp, acc in res.items():
                if exp.startswith("Top1_Acc_Exp/"):
                    curr_accpexp.append(float(acc))

        # Get Local Eval Metrics from Avalanche
        last_metrics = self.evaluation.get_last_metrics()
        confusion_matrix = last_metrics["ConfusionMatrix_Stream/eval_phase/test_stream"].tolist()
        stream_loss = last_metrics["Loss_Stream/eval_phase/test_stream"]
        stream_acc = last_metrics["Top1_Acc_Stream/eval_phase/test_stream"]
        stream_disc_usage = last_metrics["DiskUsage_Stream/eval_phase/test_stream"]

        # Calculating Forgetting Measures
        local_eval_metrics = self.client_state.config_records["local_eval_metrics"]
        hist_accpexp = local_eval_metrics["accuracy_per_exp"]

        # Calculating Running Cumalative Forgetting Measure
        cm_fmpexp = []
        for i, e in enumerate(hist_accpexp):
            e = json.loads(e)
            fm = e[i] - curr_accpexp[i]
            cm_fmpexp.append(fm)

        # Checking Cumalative Forgetting Measure
        cprint("Check Cumalative FM", "blue")
        cprint("History of Accuracy per Experience for this client")
        print(json.dumps(hist_accpexp, indent=2))
        print(f"Current Accuracy per Experience: {json.dumps(curr_accpexp, indent=4)}")
        print(f"Cumalative Forgetting per Experience: {json.dumps(cm_fmpexp, indent=4)}")
        print(f"Cumalative Forgetting Measure: {cmfm}")
 
        # Calculate Running Stepwise Forgetting Measure
        sw_fmpexp = []
        if hist_accpexp:
            prev_accpexp = json.loads(hist_accpexp[-1])
        else:
            prev_accpexp = []
        for i, (prev_acc, curr_acc) in enumerate(zip(prev_accpexp, curr_accpexp)):
            sw_fmpexp.append(prev_acc - curr_acc)
        swfm = sum(sw_fmpexp)/NUM_EXP

        # Checking Stepwise Forgetting Measure
        cprint("Check StepWise FM", "blue")
        print(f"Current Accuracy per Experience: {json.dumps(curr_accpexp, indent=4)}")
        print(f"Prev Accuracy per Experience {json.dumps(prev_accpexp, indent=4)}")
        print(f"StepWise Forgetting per Experience: {json.dumps(sw_fmpexp, indent=4)}")
        print(f"StepWise Forgetting Measure: {swfm}")
            
        # Make Fit Metrics Dictionary
        fit_dict_return = {
                "confusion_matrix": json.dumps(confusion_matrix),
                "cumalative_forgetting_measure":  float(cmfm),
                "stepwise_forgetting_measure": float(swfm),
                "stream_loss":  float(stream_loss),
                "stream_acc":  float(stream_acc),
                "stream_disc_usage":  float(stream_disc_usage),
                "accuracy_per_experience": json.dumps(curr_accpexp),
                "stepwise_forgetting_per_exp": json.dumps(sw_fmpexp),
                "cumalative_forgetting_per_exp": json.dumps(cm_fmpexp),
                "pid": self.partition_id,
                "round": rnd,
            }
        cprint("----------------------------CLIENT_INFO--------------------------------")
        print(fit_dict_return)
        cprint('-----------------------------------------------------------------------')

        
        # Logging Client State
        print("Logging Client States")
        if rnd != 0:
            metrics = {}
            metrics["accuracy_per_exp"] = [json.dumps(curr_accpexp)]
            metrics["stream_accuracy"] = [stream_acc]
            metrics["stream_loss"] = [stream_loss]
            metrics["cumalative_forgetting_measure"] = [cmfm]
            metrics["stepwise_forgetting_measure"] = [swfm]
            
            # Update existing metrics if they exist
            if "accuracy_per_exp" in local_eval_metrics:
                metrics["accuracy_per_exp"].extend(local_eval_metrics["accuracy_per_exp"])
            if "stream_accuracy" in local_eval_metrics:
                metrics["stream_accuracy"].extend(local_eval_metrics["stream_accuracy"])
            if "stream_loss" in local_eval_metrics:
                metrics["stream_loss"].extend(local_eval_metrics["stream_loss"])
            if "cumalative_forgetting_measure" in local_eval_metrics:
                metrics["cumalative_forgetting_measure"].extend(local_eval_metrics["cumalative_forgetting_measure"])
            if "stepwise_forgetting_measure" in local_eval_metrics:
                metrics["stepwise_forgetting_measure"].extend(local_eval_metrics["stepwise_forgetting_measure"])
            
            self.client_state.config_records["local_eval_metrics"] = metrics

        print("Finished Fit")
        # Client Failure Provision
        if random.random() < cfg.client.falloff:
            return None
        else:
            return get_parameters(self.cl_strategy.model), self.trainlen_per_exp[rnd-1], {}

    # Evaluate After Updating Global Model
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

# Function that launches a Client
def client_fn(context: Context) -> Client:
    """Create a Flower client representing a single organization."""

    # Load model
    net = get_model().to(DEVICE)

    # Grab Partition Data
    partition_id = context.node_config["partition-id"]
    train_data, test_data = load_datasets(partition_id=partition_id)

    total_train_samples = len(train_data)
    total_eval_samples = len(test_data)

    # Splitting Data into Experiences
    n_experiences = cfg.cl.num_experiences
    train_experiences = split_dataset(train_data, n_experiences)
    test_experiences = split_dataset(test_data, n_experiences)  # optional

    # Preparing list of train experiences
    trainlen_per_exp = []
    ava_train = []
    for i, exp in enumerate(train_experiences, start=1):
        trainlen_per_exp.append(len(exp))
        ava_exp = as_avalanche_dataset(exp)
        ava_train.append(exp)

    # Preparing list of test experiences
    testlen_per_exp = []
    ava_test = []
    for i, exp in enumerate(test_experiences):
        testlen_per_exp.append(len(exp))
        ava_exp = as_avalanche_dataset(exp)
        ava_test.append(exp)

    # Creating benchmarks 
    benchmark = benchmark_from_datasets(train=ava_train, test=ava_test)

    # Print ClientID
    print(f"---------------------------------LAUNCHING CLIENT: {partition_id}-----------------------------------------------")

    return FlowerClient(context, net, benchmark, trainlen_per_exp, testlen_per_exp, partition_id).to_client()


