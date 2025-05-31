# File: run_vehicle_experiment.py
import argparse
from pathlib import Path

# Import the integration script
from integrate_vehicle_dataset import integrate_vehicle_dataset
from data.vehicle_dataset import download_and_prepare_vehicle_dataset

# Import main experiment function from the continual learning framework
from main import run

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run continual learning with the Vehicle dataset')
    
    # Dataset arguments
    parser.add_argument('--data-dir', type=str, default='./data/vehicle-dataset-for-yolo', 
                      help='Directory to store the dataset')
    parser.add_argument('--download', action='store_true', 
                      help='Download and set up the dataset')
    
    # Experiment arguments
    parser.add_argument('--contexts', type=int, default=6, 
                      help='Number of contexts (1, 2, 3, or 6)')
    parser.add_argument('--scenario', type=str, default='task', 
                      choices=['task', 'domain', 'class'],
                      help='Continual learning scenario')
    
    # Method arguments
    parser.add_argument('--method', type=str, default='si', 
                      choices=['none', 'ewc', 'si', 'lwf', 'xdg', 'replay-generative', 
                              'replay-buffer', 'brain-inspired', 'separate-networks'],
                      help='Continual learning method to use')
    
    args = parser.parse_args()
    
    # Integrate Vehicle dataset
    # integrate_vehicle_dataset()
    
    # Download dataset if requested
    if args.download:
        download_and_prepare_vehicle_dataset(args.data_dir)
    
    # Prepare method arguments
    method_args = {}
    if args.method == 'ewc':
        method_args['ewc'] = True
    elif args.method == 'si':
        method_args['si'] = True
    elif args.method == 'lwf':
        method_args['lwf'] = True
    elif args.method == 'xdg':
        method_args['xdg'] = True
    elif args.method == 'replay-generative':
        method_args['replay'] = 'generative'
    elif args.method == 'replay-buffer':
        method_args['replay'] = 'buffer'
    elif args.method == 'brain-inspired':
        method_args['brain_inspired'] = True
    elif args.method == 'separate-networks':
        method_args['separate_networks'] = True
    
    # Run the experiment
    run(
        experiment='Vehicle',
        scenario=args.scenario,
        contexts=args.contexts,
        data_dir=args.data_dir,
        **method_args
    )

if __name__ == "__main__":
    main()
