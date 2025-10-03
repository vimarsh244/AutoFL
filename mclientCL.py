# Import From Custom Modules
from clutils.ParamFns import set_parameters, get_parameters
from clutils.make_experiences import split_dataset
from clutils.clstrat import make_cl_strat 

#Import basic Modules
import json
import random
import os
import warnings
import time
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
from utils.latency_simulator import LatencySimulator

cfg = load_config()

latency_simulator = LatencySimulator(cfg)


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

# Client Class
class FlowerClient(NumPyClient):
    def __init__(self, context: Context, net, benchmark, trainlen_per_exp, testlen_per_exp, partition_id):
        self.client_state = context.state
        if not hasattr(self.client_state, 'config_records'):
            self.client_state.config_records = ConfigRecord()
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
        self.latency_sim = latency_simulator
        self.latency_enabled = self.latency_sim.enabled and self.latency_sim.has_client(partition_id)
        self.permanent_drop = False
        self._payload_bytes = None

        # To add  later: Battery, Location, Speed, Mobility_Trace

        print(self.client_state.config_records)

    # Get Params from Global Model
    def get_parameters(self, config):
        return get_parameters(self.cl_strategy.model)

    def _ensure_payload_bytes(self, parameters) -> int:
        if self._payload_bytes is None:
            try:
                self._payload_bytes = int(sum(arr.nbytes for arr in parameters))
            except AttributeError:
                self._payload_bytes = 0
        return self._payload_bytes or 0

    # Fit on Local Data
    def fit(self, parameters, config):
        set_parameters(self.cl_strategy.model, parameters)
        rnd = config["server_round"]
        num_rounds = config["num_rounds"]

        round_start = time.time()

        cprint("FIT")
        print(f"Client {self.partition_id} Fit on round: {rnd}")

        payload_bytes = self._ensure_payload_bytes(parameters)
        latency_sample = None
        base_delay_s = 0.0
        download_time_s = 0.0
        upload_time_s = 0.0
        expected_network_time_s = 0.0
        threshold_s = float("inf")
        exceeded_threshold = False

        if self.latency_enabled:
            latency_sample = self.latency_sim.sample(self.partition_id, rnd, payload_bytes)
            base_delay_s = latency_sample.base_delay_s
            download_time_s = latency_sample.download_time_s
            upload_time_s = latency_sample.upload_time_s
            expected_network_time_s = latency_sample.total_network_time_s
            threshold_s = latency_sample.threshold_s
            exceeded_threshold = latency_sample.exceeded_threshold
            if exceeded_threshold and not self.permanent_drop:
                cprint(
                    f"Client {self.partition_id} latency sample {expected_network_time_s:.3f}s exceeds threshold {threshold_s:.3f}s",
                    "yellow",
                )
            self.latency_sim.sleep_pre_training(latency_sample)

        drop_due_to_latency = self.permanent_drop or (self.latency_enabled and exceeded_threshold)
        if drop_due_to_latency and not self.permanent_drop and self.latency_sim.should_remove_permanently():
            cprint(f"Client {self.partition_id} marked for permanent removal due to latency", "red")
            self.permanent_drop = True

        results = []
        
        # Handle different benchmark types
        if hasattr(self.benchmark, 'train_stream'):
            train_stream = self.benchmark.train_stream
        elif hasattr(self.benchmark, 'train_datasets_stream'): # Might not need
            train_stream = self.benchmark.train_datasets_stream
        else:
            raise ValueError(f"Unknown benchmark type: {type(self.benchmark)}")
            
        # Calculate which experience to train on (cycle through available experiences)
        experience_idx = ((rnd - 1) % NUM_EXP)
        print(f"Round {rnd}: Training on experience {experience_idx} (cycling through {len(self.trainlen_per_exp)} experiences)")
        
        training_start = time.time()
        training_duration = 0.0
        if not drop_due_to_latency:
            for i, experience in enumerate(train_stream):
                if i == experience_idx:
                    print(f"EXP: {experience.current_experience}")
                    trainres = self.cl_strategy.train(experience)
                    cprint('Training completed: ')
                    break  # Only train on current experience
            training_duration = time.time() - training_start
        else:
            cprint(f"Skipping training for client {self.partition_id} due to latency threshold", "yellow")
            training_duration = 0.0

        # Local Eval after fit on client for metrics
        evaluation_duration = 0.0
        if not drop_due_to_latency:
            eval_start = time.time()
            print(f"Local Evaluation of client {self.partition_id} on round {rnd}")
            results.append(self.cl_strategy.eval(self.benchmark.test_stream))
            evaluation_duration = time.time() - eval_start

        local_eval_metrics = self.client_state.config_records["local_eval_metrics"]

        if drop_due_to_latency:
            curr_accpexp = []
            stream_loss = 0.0
            stream_acc = 0.0
            cm_fmpexp = []
            sw_fmpexp = []
            cmfm = 0.0
            swfm = 0.0
        else:
            # Calc Accuracy per Experience 
            curr_accpexp = []
            for res in results:
                for exp, acc in res.items():
                    if exp.startswith("Top1_Acc_Exp/"):
                        curr_accpexp.append(float(acc))

            # Get Local Eval Metrics from Avalanche
            last_metrics = self.evaluation.get_last_metrics()
            print("DEBUG: Available metrics keys:", list(last_metrics.keys()))  # Debug print
            
            # Handle different stream naming conventions
            stream_suffix = "/eval_phase/test_stream"

            stream_loss = last_metrics.get(f"Loss_Stream{stream_suffix}", 0.0)
            stream_acc = last_metrics.get(f"Top1_Acc_Stream{stream_suffix}", 0.0)

            # Calculating Forgetting Measures
            hist_accpexp = local_eval_metrics["accuracy_per_exp"]

            cm_fmpexp = []
            if curr_accpexp:
                for i, e in enumerate(hist_accpexp):
                    e = json.loads(e)
                    fm = e[i % NUM_EXP] - curr_accpexp[i % NUM_EXP]
                    cm_fmpexp.append(fm)
            if cm_fmpexp:
                cmfm = sum(cm_fmpexp)/len(cm_fmpexp)
            else: 
                cmfm = 0.0

            cprint("Check Cumalative FM", "blue")
            cprint("History of Accuracy per Experience for this client")
            print(json.dumps(hist_accpexp, indent=2))
            print(f"Current Accuracy per Experience: {json.dumps(curr_accpexp, indent=4)}")
            print(f"Cumalative Forgetting per Experience: {json.dumps(cm_fmpexp, indent=4)}")
            print(f"Cumalative Forgetting Measure: {cmfm}")
 
            sw_fmpexp = []
            if hist_accpexp and curr_accpexp:
                prev_accpexp = json.loads(hist_accpexp[-1]) if hist_accpexp else []
                for prev_acc, curr_acc in zip(prev_accpexp, curr_accpexp):
                    sw_fmpexp.append(prev_acc - curr_acc)
            if sw_fmpexp:
                swfm = sum(sw_fmpexp)/NUM_EXP 
            else: 
                swfm = 0.0

            cprint("Check StepWise FM", "blue")
            print(f"Current Accuracy per Experience: {json.dumps(curr_accpexp, indent=4)}")
            prev_accpexp = json.loads(hist_accpexp[-1]) if hist_accpexp else []
            print(f"Prev Accuracy per Experience {json.dumps(prev_accpexp, indent=4)}")
            print(f"StepWise Forgetting per Experience: {json.dumps(sw_fmpexp, indent=4)}")
            print(f"StepWise Forgetting Measure: {swfm}")
            
        # Make Fit Metrics Dictionary
        total_round_time = time.time() - round_start

        network_sleep_after = 0.0
        if latency_sample is not None and not drop_due_to_latency:
            network_sleep_after = latency_sample.upload_time_s
            self.latency_sim.sleep_post_training(latency_sample)

        fit_dict_return = {
         #       "confusion_matrix": json.dumps(confusion_matrix),  # Disabled for now
                "cumalative_forgetting_measure":  float(cmfm),
                "stepwise_forgetting_measure": float(swfm),
                "stream_loss":  float(stream_loss),
                "stream_acc":  float(stream_acc),
                "stream_disc_usage":  float(0.0),
                "accuracy_per_experience": json.dumps(curr_accpexp),
                "stepwise_forgetting_per_exp": json.dumps(sw_fmpexp),
                "cumalative_forgetting_per_exp": json.dumps(cm_fmpexp),
                "pid": self.partition_id,
                "round": rnd,
                "latency/enabled": bool(self.latency_enabled),
                "latency/base_delay_s": float(base_delay_s),
                "latency/download_time_s": float(download_time_s),
                "latency/upload_time_s": float(upload_time_s),
                "latency/expected_network_time_s": float(expected_network_time_s),
                "latency/threshold_s": float(threshold_s),
                "latency/dropped": bool(drop_due_to_latency),
                "latency/upload_sleep_s": float(network_sleep_after),
                "timing/training_s": float(training_duration),
                "timing/evaluation_s": float(evaluation_duration),
                "timing/round_total_s": float(total_round_time),
            }
        cprint("----------------------------Results After Fit--------------------------------")
        print(json.dumps(fit_dict_return, indent=4))
        cprint('-----------------------------------------------------------------------')

        
        # Logging Client State
        print("Logging Client States")
        if rnd != 0 and not drop_due_to_latency:
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
            
        cprint("Finished Fit")
        
        # MEMORY CLEANUP - clear CUDA cache and run garbage collection
        clear_memory()
        print(f"Memory cleared after fit round {rnd}")
        
        # Client Failure Provision
        if random.random() < cfg.client.falloff:
            return None
        else:
            # Use the same cycling logic for experience length
            experience_idx = ((rnd - 1) % NUM_EXP)
            return get_parameters(self.cl_strategy.model), self.trainlen_per_exp[experience_idx], fit_dict_return

    # Evaluate After Updating Global Model
    def evaluate(self, parameters, config):
        # Setting Global Model param
        set_parameters(self.net, parameters)
        rnd = config["server_round"]
        num_rounds = config["num_rounds"]

        # Creating a new CL Strategy for Evaluation
        cl_strategy, evaluation = make_cl_strat(self.net)

        # Distributed Client Evaluation
        results = []
        print(f"------------------------Local Client {self.partition_id} Evaluation on Updated Global Model--------------------")
        
        # Handle different benchmark types
        if hasattr(self.benchmark, 'test_stream'):
            test_stream = self.benchmark.test_stream
        else:
            test_stream = self.benchmark.test_datasets_stream
        
        results.append(cl_strategy.eval(test_stream))
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

        cm_fmpexp = []
        for i, e in enumerate(hist_accpexp):
            e = json.loads(e)
            fm = e[i % NUM_EXP] - curr_accpexp[i % NUM_EXP];
            cm_fmpexp.append(fm)
        if cm_fmpexp:
            cmfm = sum(cm_fmpexp)/len(cm_fmpexp)
        else:
            cmfm = 0

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


        print("Eval of Client: ")
        print("Loss: ", stream_loss)
        print("Acc: ", stream_acc)
        print("Per Exp Acc: ", curr_accpexp)

        eval_dict_return = {
                "stream_accuracy": float(stream_acc),
                "stream_loss": float(stream_loss),
                "accuracy_per_experience": json.dumps(curr_accpexp),
                "stepwise_forgetting_measure": float(swfm),  # not calculated in eval
                "cumalative_forgetting_measure": float(cmfm),  # not calculated in eval
                "stepwise_forgetting_per_experience": json.dumps(sw_fmpexp),  # not calculated in eval
                "cumalative_forgetting_per_experience": json.dumps(cm_fmpexp),  # not calculated in eval
                "server_round": rnd,
                "pid": self.partition_id,
                }

        # Printing Global Evaluation Results:
        print(f"Global Distributed Evaluation of Client {self.partition_id}")
        print(json.dumps(eval_dict_return, indent=4))

        cprint("Logging Client States")
        # Note: global evaluation metrics logging disabled for now
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
        elif hasattr(benchmark, 'train_datasets_stream'):
            # CLScenario object
            trainlen_per_exp = [len(exp.dataset) for exp in benchmark.train_datasets_stream]
            testlen_per_exp = [len(exp.dataset) for exp in benchmark.test_datasets_stream]
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
        elif hasattr(benchmark, 'train_datasets_stream'):
            # CLScenario object
            trainlen_per_exp = [len(exp.dataset) for exp in benchmark.train_datasets_stream]
            testlen_per_exp = [len(exp.dataset) for exp in benchmark.test_datasets_stream]
        else:
            raise ValueError(f"Unknown benchmark type: {type(benchmark)}")

    # Print ClientID
    print("------------------------------------------------ClientID: ", partition_id, "----------------------------------------------")

    # Create a single Flower client representing a single organization
    # FlowerClient is a subclass of NumPyClient, so we need to call .to_client()
    # to convert it to a subclass of `flwr.client.Client`
    return FlowerClient(context, net, benchmark, trainlen_per_exp, testlen_per_exp, partition_id).to_client()


