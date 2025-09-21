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
import gc  # For garbage collection
from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Metrics, Context, ConfigRecord

# Ignore Flower Warnings
warnings.filterwarnings("ignore")

#Setting up Configuration
from config_utils import load_config
cfg = load_config()

# Optional availability import for client-side guard
try:
    from sim.availability import load_scheduler_from_csvs
except Exception:
    load_scheduler_from_csvs = None


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
elif cfg.dataset.workload == "bdd100k_v2":
    from workloads.BDD100KDomainCLV2 import load_datasets
elif cfg.dataset.workload == "kitti_v2":
    from workloads.KITTIDomainCLV2 import load_datasets
elif cfg.dataset.workload == "bdd100k_10k":
    from workloads.BDD100K10kDomainCL import load_datasets
elif cfg.dataset.workload == "permuted_mnist":
    from workloads.PermutedMNIST import load_datasets
elif cfg.dataset.workload == "rotated_mnist":
    from workloads.RotatedMNIST import load_datasets
elif cfg.dataset.workload == "mnist":
    from workloads.MNIST import load_datasets
elif cfg.dataset.workload == "split_cifar10":
    from workloads.SplitCIFAR10 import load_datasets
elif cfg.dataset.workload == "split_cifar100":
    from workloads.SplitCIFAR100 import load_datasets
elif cfg.dataset.workload == "core50":
    from workloads.CORe50 import load_datasets
else:
    raise ValueError(f"Unknown workload: {cfg.dataset.workload}")

# Setting Global Variables
# Respect configuration: only use GPU if num_gpus > 0.0 AND CUDA is available
if cfg.client.num_gpus > 0.0 and torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
    print(f"Using GPU: {DEVICE}")
else:
    DEVICE = torch.device("cpu")
    print(f"Using CPU: {DEVICE}")
    
BATCH_SIZE = cfg.dataset.batch_size
NUM_CLIENTS = cfg.server.num_clients
NUM_EXP = cfg.cl.num_experiences

# Prepare shared scheduler (optional) for client-side behavior
_client_scheduler = None
if getattr(cfg, 'sim', None) and getattr(cfg.sim, 'enabled', False) and load_scheduler_from_csvs is not None:
    try:
        _client_scheduler = load_scheduler_from_csvs(
            csv_paths=[str(p) for p in getattr(cfg.sim, 'csv_files', [])],
            signal=getattr(cfg.sim, 'signal', 'transmissionState'),
            threshold=getattr(cfg.sim, 'threshold', None),
            time_scale=getattr(cfg.sim, 'time_scale', 1.0),
            round_duration_s=getattr(cfg.sim, 'round_duration_s', 1.0),
            start_time_s=getattr(cfg.sim, 'start_time_s', 0.0),
            randomize_mapping=getattr(cfg.sim, 'randomize_mapping', False),
            num_clients=cfg.server.num_clients,
            rnd_seed=getattr(cfg.sim, 'seed', 42),
            min_gap_s=getattr(cfg.sim, 'min_gap_s', 0.0),
        )
        print("[Client] Availability scheduler loaded")
    except Exception as e:
        print(f"[Client] Failed to load scheduler: {e}")
        _client_scheduler = None

# Color print function
def clear_memory():
    """Clear CUDA memory and run garbage collection"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

def cprint(text, color="green"):
    """Print text with color. Available colors: red, green, yellow, blue, magenta, cyan, white"""
    colors = {
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'magenta': '\033[95m',
        'cyan': '\033[96m',
        'white': '\033[97m',
        'reset': '\033[0m'
    }
    color_code = colors.get(color.lower(), colors['green'])
    print(f"{color_code}{text}{colors['reset']}")

def get_model():
    """Get model based on configuration"""
    # use intelligent model factory
    from utils.model_factory import create_model
    return create_model(cfg)

# Persistent State of Clients
partition_strategies = [make_cl_strat(get_model().to(DEVICE)) for _ in range(NUM_CLIENTS)]

# Set client_id for FedWeIT strategies
for client_id, (strategy, evaluation) in enumerate(partition_strategies):
    if hasattr(strategy, 'current_client_id'):
        strategy.current_client_id = client_id
        print(f"Set client_id {client_id} for FedWeIT strategy")


# Client Class
class FlowerClient(NumPyClient):
    def __init__(self, context: Context, net, benchmark, trainlen_per_exp, testlen_per_exp, partition_id):
        self.client_state = (context.state)
        if "local_eval_metrics" not in self.client_state.config_records:
            self.client_state.config_records["local_eval_metrics"] = ConfigRecord()
        if "global_eval_metrics" not in self.client_state.config_records:
            self.client_state.config_records["global_eval_metrics"] = ConfigRecord()
        if "availability" not in self.client_state.config_records:
            self.client_state.config_records["availability"] = ConfigRecord()
        # Special Provision for acc per exp as needed to calculate fm
        if "accuracy_per_exp" not in self.client_state.config_records["local_eval_metrics"]:
            self.client_state.config_records["local_eval_metrics"]["accuracy_per_exp"] = []
        if "accuracy_per_exp" not in self.client_state.config_records["global_eval_metrics"]:
            self.client_state.config_records["global_eval_metrics"]["accuracy_per_exp"] = []
        if "rounds_selected" not in self.client_state.config_records["local_eval_metrics"]:
            self.client_state.config_records["local_eval_metrics"]["rounds_selected"] = []
        if "rounds_selected" not in self.client_state.config_records["global_eval_metrics"]:
            self.client_state.config_records["global_eval_metrics"]["rounds_selected"] = []
        self.net = net
        self.benchmark = benchmark
        self.trainlen_per_exp = trainlen_per_exp
        self.testlen_per_exp = testlen_per_exp
        self.cl_strategy, self.evaluation = partition_strategies[partition_id]
        self.partition_id = partition_id

        # To add  later: Battery, Location, Speed, Mobility_Trace

        print(self.client_state.config_records)

    def _is_available(self, rnd: int) -> bool:
        if _client_scheduler is None:
            return True
        return _client_scheduler.is_available(self.partition_id, rnd)

    def _log_availability(self, rnd: int, phase: str, state: str):
        try:
            av = self.client_state.config_records["availability"]
            key = f"{phase}_round_{rnd}"
            if key not in av:
                av[key] = []
            av[key].append({"client": self.partition_id, "state": state})
        except Exception:
            pass

    # Get Params from Global Model
    def get_parameters(self, config):
        return get_parameters(self.cl_strategy.model)

    # Fit on Local Data
    def fit(self, parameters, config):
        set_parameters(self.cl_strategy.model, parameters)
        rnd = config["server_round"]
        num_rounds = config["num_rounds"]

        # Availability check before training
        if not self._is_available(rnd):
            self._log_availability(rnd, "fit", "unavailable")
            # Simulate dropout by returning None (Flower will handle failures)
            return None
        self._log_availability(rnd, "fit", "start")

        cprint("FIT")
        print(f"Client {self.partition_id} Fit on round: {rnd}")

        # Random mid-round drop simulation (client.falloff)
        drop_mid = random.random() < cfg.client.falloff

        # Determine which experience to train this round
        prog = getattr(cfg.cl, 'progression', None)
        mode = getattr(prog, 'mode', 'per_round') if prog is not None else 'per_round'
        if mode == 'per_k_rounds':
            k = int(getattr(prog, 'k', 1)) or 1
            target_idx = int(((rnd - 1) // k) + 1)
        else:  # per_round
            target_idx = rnd

        # Clamp to available experiences
        max_idx = len(self.benchmark.train_stream)
        if target_idx > max_idx:
            target_idx = max_idx

        # Train on selected Experience
        cprint("Starting Training")
        results = []
        for i, experience in enumerate(self.benchmark.train_stream, start=1):
            if i == target_idx:
                print(f"EXP: {experience.current_experience}")
                if drop_mid and random.random() < 0.5:
                    self._log_availability(rnd, "fit", "mid_drop")
                    return None
                trainres = self.cl_strategy.train(experience)
                cprint('Training completed: ')
                break

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
        # confusion_matrix = last_metrics["ConfusionMatrix_Stream/eval_phase/test_stream"].tolist()
        stream_loss = last_metrics["Loss_Stream/eval_phase/test_stream"]
        stream_acc = last_metrics["Top1_Acc_Stream/eval_phase/test_stream"]
        stream_disc_usage = last_metrics["DiskUsage_Stream/eval_phase/test_stream"]

        # Calculating Forgetting Measures
        local_eval_metrics = self.client_state.config_records["local_eval_metrics"]
        hist_accpexp = local_eval_metrics["accuracy_per_exp"]
        round_fit = local_eval_metrics["rounds_selected"]

        # Calculating Running Cumalative Forgetting Measure
        cm_fmpexp = []
        for i, e in enumerate(hist_accpexp):
            e = json.loads(e)
            fm = e[i] - curr_accpexp[i];
            cm_fmpexp.append(fm)
        if cm_fmpexp:
            cmfm = sum(cm_fmpexp)/len(cm_fmpexp)
        else:
            cmfm = 0

        # Calculate Running Stepwise Forgetting Measure
        sw_fmpexp = []
        if hist_accpexp:
            prev_accpexp = json.loads(hist_accpexp[-1])
        else:
            prev_accpexp = []
        for i, (prev_acc, curr_acc) in enumerate(zip(prev_accpexp, curr_accpexp)):
            sw_fmpexp.append(prev_acc - curr_acc)
        swfm = sum(sw_fmpexp)/NUM_EXP
            
        # Make Fit Metrics Dictionary
        fit_dict_return = {
                # "confusion_matrix": json.dumps(confusion_matrix),
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

        # Logging Client State
        if rnd != 0:
            if "accuracy_per_exp" not in local_eval_metrics:
                local_eval_metrics["accuracy_per_exp"] = [json.dumps(curr_accpexp)]
            else:
                local_eval_metrics["accuracy_per_exp"].append(json.dumps(curr_accpexp))
            if "stream_accuracy" not in local_eval_metrics:
                local_eval_metrics["stream_accuracy"] = [stream_acc]
            else:
                local_eval_metrics["stream_accuracy"].append(stream_acc)
            if "stream_loss" not in local_eval_metrics:
                local_eval_metrics["stream_loss"] = [stream_loss]
            else:
                local_eval_metrics["stream_loss"].append(stream_loss)
            if "cumalative_forgetting_measure" not in local_eval_metrics:
                local_eval_metrics["cumalative_forgetting_measure"] = [cmfm] 
            else:
                local_eval_metrics["cumalative_forgetting_measure"].append(cmfm)
            if "stepwise_forgetting_measure" not in local_eval_metrics:
                local_eval_metrics["stepwise_forgetting_measure"] = [swfm]
            else:
                local_eval_metrics["stepwise_forgetting_measure"].append(swfm)
            local_eval_metrics["rounds_selected"].append(rnd)

        # Finish
        if random.random() < cfg.client.falloff:
            self._log_availability(rnd, "fit", "end_drop")
            return None
        else:
            self._log_availability(rnd, "fit", "end")
            return get_parameters(self.cl_strategy.model), self.trainlen_per_exp[rnd-1], fit_dict_return

    # Evaluate After Updating Global Model
    def evaluate(self, parameters, config):
        # Setting Global Model param
        set_parameters(self.net, parameters)
        rnd = config["server_round"]
        num_rounds = config["num_rounds"]

        if not self._is_available(rnd):
            self._log_availability(rnd, "eval", "unavailable")
            return None

        # Creating a new CL Strategy for Evaluation
        cl_strategy, evaluation = make_cl_strat(self.net)

        # Distributed Client Evaluation
        results = []
        print(f"------------------------Local Client {self.partition_id} Evaluation on Updated Global Model--------------------")
        results.append(cl_strategy.eval(self.benchmark.test_stream))
        last_metrics = evaluation.get_last_metrics()
        stream_loss = last_metrics["Loss_Stream/eval_phase/test_stream"]
        stream_acc = last_metrics["Top1_Acc_Stream/eval_phase/test_stream"]

        # Getting Accuracy per Experience for client
        curr_accpexp = []
        for res in results:
            for exp, acc in res.items():
                if exp.startswith("Top1_Acc_Exp/"):
                    curr_accpexp.append(float(acc))
                    
         # Calculating Forgetting Measures
        global_eval_metrics = self.client_state.config_records["global_eval_metrics"]
        hist_accpexp = global_eval_metrics["accuracy_per_exp"]

        # Calculating Running Cumalative Forgetting Measure
        cm_fmpexp = []
        for i, e in enumerate(hist_accpexp):
            e = json.loads(e)
            fm = e[i] - curr_accpexp[i];
            cm_fmpexp.append(fm)
        if cm_fmpexp:
            cmfm = sum(cm_fmpexp)/len(cm_fmpexp)
        else:
            cmfm = 0

        # Calculate Running Stepwise Forgetting Measure
        sw_fmpexp = []
        if hist_accpexp:
            prev_accpexp = json.loads(hist_accpexp[-1])
        else:
            prev_accpexp = []
        for i, (prev_acc, curr_acc) in enumerate(zip(prev_accpexp, curr_accpexp)):
            sw_fmpexp.append(prev_acc - curr_acc)
        swfm = sum(sw_fmpexp)/NUM_EXP

        eval_dict_return = {
                "stream_accuracy": float(stream_acc),
                "stream_loss": float(stream_loss),
                "accuracy_per_experience": json.dumps(curr_accpexp),
                "stepwise_forgetting_measure": float(swfm),
                "cumalative_forgetting_measure":  float(cmfm),
                "stepwise_forgetting_per_experience": json.dumps(sw_fmpexp),
                "cumalative_forgetting_per_experience": json.dumps(cm_fmpexp),
                "server_round": rnd,
                "pid": self.partition_id,
                }

        # Logging Client States
        if rnd != 0:
            if "accuracy_per_exp" not in global_eval_metrics:
                global_eval_metrics["accuracy_per_exp"] = [json.dumps(curr_accpexp)]
            else:
                global_eval_metrics["accuracy_per_exp"].append(json.dumps(curr_accpexp))
            if "stream_accuracy" not in global_eval_metrics:
                global_eval_metrics["stream_accuracy"] = [stream_acc]
            else:
                global_eval_metrics["stream_accuracy"].append(stream_acc)
            if "stream_loss" not in global_eval_metrics:
                global_eval_metrics["stream_loss"] = [stream_loss]
            else:
                global_eval_metrics["stream_loss"].append(stream_loss)
            if "cumalative_forgetting_measure" not in global_eval_metrics:
                global_eval_metrics["cumalative_forgetting_measure"] = [cmfm] 
            else:
                global_eval_metrics["cumalative_forgetting_measure"].append(cmfm)
            if "stepwise_forgetting_measure" not in global_eval_metrics:
                global_eval_metrics["stepwise_forgetting_measure"] = [swfm]
            else:
                global_eval_metrics["stepwise_forgetting_measure"].append(swfm)
            global_eval_metrics["rounds_selected"].append(rnd)

        return float(stream_loss), sum(self.testlen_per_exp), eval_dict_return


# Function that launches a Client
def client_fn(context: Context) -> Client:
    """Create a Flower client representing a single organization."""

    # Load model
    net = get_model().to(DEVICE)

    # Grab Partition Data
    partition_id = context.node_config["partition-id"]

    # load_datasets may return a tuple (train_data, test_data) or a benchmark object
    dataset_result = load_datasets(partition_id=partition_id)

    if isinstance(dataset_result, tuple):
        # Regular CL: (train_data, test_data)
        train_data, test_data = dataset_result
        n_experiences = cfg.cl.num_experiences
        train_experiences = split_dataset(train_data, n_experiences)
        test_experiences = split_dataset(test_data, n_experiences)
        trainlen_per_exp = [len(exp) for exp in train_experiences]
        testlen_per_exp = [len(exp) for exp in test_experiences]
        from avalanche.benchmarks.scenarios.dataset_scenario import benchmark_from_datasets
        benchmark = benchmark_from_datasets(train=train_experiences, test=test_experiences)
    elif isinstance(dataset_result, dict):
        # CORe50 or other workloads that return a dictionary with benchmark and metadata
        benchmark = dataset_result["benchmark"]
        n_experiences = cfg.cl.num_experiences
        
        # Handle CLScenario objects
        if hasattr(benchmark, 'train_stream'):
            # Standard benchmark
            trainlen_per_exp = [len(exp.dataset) for exp in benchmark.train_stream]
            testlen_per_exp = [len(exp.dataset) for exp in benchmark.test_stream]
        else:
            raise ValueError(f"Unknown benchmark type: {type(benchmark)}")
    else:
        # DomainCL: benchmark object
        benchmark = dataset_result
        n_experiences = cfg.cl.num_experiences
        
        # Handle CLScenario objects
        if hasattr(benchmark, 'train_stream'):
            # Standard benchmark
            trainlen_per_exp = [len(exp.dataset) for exp in benchmark.train_stream]
            testlen_per_exp = [len(exp.dataset) for exp in benchmark.test_stream]
        else:
            raise ValueError(f"Unknown benchmark type: {type(benchmark)}")

    # Print ClientID
    print("ClientID: ", partition_id)

    # Create a single Flower client representing a single organization
    # FlowerClient is a subclass of NumPyClient, so we need to call .to_client()
    # to convert it to a subclass of `flwr.client.Client`
    return FlowerClient(context, net, benchmark, trainlen_per_exp, testlen_per_exp, partition_id).to_client()


