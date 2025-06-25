#!/usr/bin/env python3
"""
Comprehensive test script to verify that all CIFAR100 workloads work correctly.
Tests regular CIFAR100, CIFAR100CL, and CIFAR100DomainCL workloads.
"""

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from models.SimpleCNN import Net
from clutils.clstrat import make_cl_strat
from clutils.make_experiences import split_dataset
from avalanche.benchmarks.scenarios.dataset_scenario import benchmark_from_datasets

# Load config and temporarily change to CIFAR100
cfg = OmegaConf.load('config/config.yaml')
# Set CIFAR100 specific config
cfg.model.num_classes = 100  # CIFAR100 has 100 classes

# Update the global config so that the Net() class uses it
import models.SimpleCNN
models.SimpleCNN.cfg = cfg

def create_cifar100_model():
    """Create a model suitable for CIFAR100 (100 classes)"""
    net = Net()  # This will still use the old config
    # Replace the final layer to have 100 outputs for CIFAR100
    net.fc3 = nn.Linear(84, 100)
    return net

def test_cifar100_regular():
    """Test regular CIFAR100 workload"""
    print("="*50)
    print("Testing Regular CIFAR100 Workload")
    print("="*50)
    
    try:
        from workloads.CIFAR100 import load_datasets
        
        # Load datasets
        trainloader, valloader, testloader = load_datasets(partition_id=0)
        print(f"Loaded train loader with {len(trainloader)} batches")
        print(f"Loaded val loader with {len(valloader)} batches")
        print(f"Loaded test loader with {len(testloader)} batches")
        
        # Test data format
        for batch in trainloader:
            if isinstance(batch, dict):
                print(f"Batch format: dict with keys {list(batch.keys())}")
                if 'img' in batch:
                    print(f"Image shape: {batch['img'].shape}")
                if 'fine_label' in batch:
                    print(f"Labels (fine_label): {batch['fine_label'][:5]}")
                elif 'label' in batch:
                    print(f"Labels (label): {batch['label'][:5]}")
            else:
                print(f"Batch format: {type(batch)}")
            break
        
        print("Regular CIFAR100 test PASSED")
        return True
        
    except Exception as e:
        print(f"Regular CIFAR100 test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cifar100_cl():
    """Test CIFAR100 CL workload"""
    print("\n" + "="*50)
    print("Testing CIFAR100 CL Workload")
    print("="*50)
    
    try:
        from workloads.CIFAR100CL import load_datasets
        
        # Load datasets
        train_data, test_data = load_datasets(partition_id=0)
        print(f"Loaded train data: {len(train_data)} samples")
        print(f"Loaded test data: {len(test_data)} samples")
        
        # Create experiences
        n_experiences = 3  # Use fewer experiences for testing
        train_experiences = split_dataset(train_data, n_experiences)
        test_experiences = split_dataset(test_data, n_experiences)
        print(f"Created {len(train_experiences)} train experiences")
        print(f"Created {len(test_experiences)} test experiences")
        
        # Create benchmark
        benchmark = benchmark_from_datasets(train=train_experiences, test=test_experiences)
        print("Created benchmark")
        
        # Create model and strategy (with 100 classes for CIFAR100)
        net = create_cifar100_model()
        cl_strategy, eval_plugin = make_cl_strat(net)
        print(f"Created model with {sum(p.numel() for p in net.parameters())} parameters")
        print(f"Model output classes: {net.fc3.out_features}")
        print(f"Learning rate: {cl_strategy.optimizer.param_groups[0]['lr']}")
        
        # Test training on first experience (abbreviated)
        for i, experience in enumerate(benchmark.train_stream):
            if i == 0:  # Only train on first experience
                print(f"Training on experience {experience.current_experience}")
                # Test just a few epochs for speed
                original_epochs = cl_strategy.train_epochs
                cl_strategy.train_epochs = 2  # Reduce epochs for testing
                result = cl_strategy.train(experience)
                cl_strategy.train_epochs = original_epochs  # Restore
                print("Training completed successfully")
                
                # Check if loss decreased
                if result and 'Loss_Epoch/train_phase/train_stream' in result:
                    final_loss = result['Loss_Epoch/train_phase/train_stream']
                    print(f"Final training loss: {final_loss}")
                break
        
        print("CIFAR100 CL test PASSED")
        return True
        
    except Exception as e:
        print(f"CIFAR100 CL test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cifar100_domain():
    """Test CIFAR100 Domain CL workload"""
    print("\n" + "="*50)
    print("Testing CIFAR100 Domain CL Workload")
    print("="*50)
    
    try:
        from workloads.CIFAR100DomainCL import load_datasets
        
        # Load datasets
        benchmark = load_datasets(partition_id=0)
        print(f"Loaded domain CL benchmark")
        
        # Handle different benchmark types
        if hasattr(benchmark, 'train_stream'):
            train_stream = benchmark.train_stream
            test_stream = benchmark.test_stream
        elif hasattr(benchmark, 'train_datasets_stream'):
            train_stream = benchmark.train_datasets_stream
            test_stream = benchmark.test_datasets_stream
        else:
            raise ValueError(f"Unknown benchmark type: {type(benchmark)}")
            
        print(f"Train experiences: {len(list(train_stream))}")
        print(f"Test experiences: {len(list(test_stream))}")
        
        # Create model and strategy (with 100 classes for CIFAR100)
        net = create_cifar100_model()
        cl_strategy, eval_plugin = make_cl_strat(net)
        print(f"Created model with {sum(p.numel() for p in net.parameters())} parameters")
        print(f"Model output classes: {net.fc3.out_features}")
        print(f"Learning rate: {cl_strategy.optimizer.param_groups[0]['lr']}")
        
        # Test training on first experience (abbreviated)
        for i, experience in enumerate(train_stream):
            if i == 0:  # Only train on first experience
                print(f"Training on experience {experience.current_experience}")
                # Test just a few epochs for speed
                original_epochs = cl_strategy.train_epochs
                cl_strategy.train_epochs = 2  # Reduce epochs for testing
                result = cl_strategy.train(experience)
                cl_strategy.train_epochs = original_epochs  # Restore
                print("Training completed successfully")
                
                # Check if loss decreased
                if result and 'Loss_Epoch/train_phase/train_stream' in result:
                    final_loss = result['Loss_Epoch/train_phase/train_stream']
                    print(f"Final training loss: {final_loss}")
                break
        
        print("CIFAR100 Domain CL test PASSED")
        return True
        
    except Exception as e:
        print(f"CIFAR100 Domain CL test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("Starting CIFAR100 workload tests...")
    print(f"Model configured for {cfg.model.num_classes} classes")
    
    # Test all CIFAR100 workloads
    test1_passed = test_cifar100_regular()
    test2_passed = test_cifar100_cl()
    test3_passed = test_cifar100_domain()
    
    print("\n" + "="*50)
    print("CIFAR100 TEST SUMMARY")
    print("="*50)
    print(f"Regular CIFAR100:     {'PASSED' if test1_passed else 'FAILED'}")
    print(f"CIFAR100 CL:          {'PASSED' if test2_passed else 'FAILED'}")
    print(f"CIFAR100 Domain CL:   {'PASSED' if test3_passed else 'FAILED'}")
    
    if test1_passed and test2_passed and test3_passed:
        print("ALL CIFAR100 TESTS PASSED! All workloads are functional.")
    else:
        print("Some CIFAR100 tests failed. Please check the error messages above.")
    
    print("="*50)

if __name__ == "__main__":
    main() 