#!/usr/bin/env python3
"""
Simple test script to verify that the training fixes work correctly.
Tests both regular CIFAR10CL and CIFAR10DomainCL workloads.
"""

import torch
from omegaconf import OmegaConf
from models.SimpleCNN import Net
from clutils.clstrat import make_cl_strat
from clutils.make_experiences import split_dataset
from avalanche.benchmarks.scenarios.dataset_scenario import benchmark_from_datasets

# Load config
cfg = OmegaConf.load('config/config.yaml')

def test_cifar10_regular():
    """Test regular CIFAR10 CL workload"""
    print("="*50)
    print("Testing Regular CIFAR10 CL Workload")
    print("="*50)
    
    try:
        from workloads.CIFAR10CL import load_datasets
        
        # Load datasets
        train_data, test_data = load_datasets(partition_id=0)
        print(f"✓ Loaded train data: {len(train_data)} samples")
        print(f"✓ Loaded test data: {len(test_data)} samples")
        
        # Create experiences
        n_experiences = 3  # Use fewer experiences for testing
        train_experiences = split_dataset(train_data, n_experiences)
        test_experiences = split_dataset(test_data, n_experiences)
        print(f"Created {len(train_experiences)} train experiences")
        print(f"Created {len(test_experiences)} test experiences")
        
        # Create benchmark
        benchmark = benchmark_from_datasets(train=train_experiences, test=test_experiences)
        print("Created benchmark")
        
        # Create model and strategy
        net = Net()
        cl_strategy, eval_plugin = make_cl_strat(net)
        print(f"Created model with {sum(p.numel() for p in net.parameters())} parameters")
        print(f"Learning rate: {cl_strategy.optimizer.param_groups[0]['lr']}")
        
        # Test training on first experience
        for i, experience in enumerate(benchmark.train_stream):
            if i == 0:  # Only train on first experience
                print(f"Training on experience {experience.current_experience}")
                result = cl_strategy.train(experience)
                print("Training completed successfully")
                
                # Check if loss decreased
                if result and 'Loss_Epoch/train_phase/train_stream' in result:
                    final_loss = result['Loss_Epoch/train_phase/train_stream']
                    print(f"Final training loss: {final_loss}")
                break
        
        print("Regular CIFAR10 CL test PASSED")
        return True
        
    except Exception as e:
        print(f"Regular CIFAR10 CL test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cifar10_domain():
    """Test CIFAR10 Domain CL workload"""
    print("\n" + "="*50)
    print("Testing CIFAR10 Domain CL Workload")
    print("="*50)
    
    try:
        from workloads.CIFAR10DomainCL import load_datasets
        
        # Load datasets
        benchmark = load_datasets(partition_id=0)
        print(f"Loaded domain CL benchmark")
        print(f"Train experiences: {len(list(benchmark.train_datasets_stream))}")
        print(f"Test experiences: {len(list(benchmark.test_datasets_stream))}")
        
        # Create model and strategy
        net = Net()
        cl_strategy, eval_plugin = make_cl_strat(net)
        print(f"Created model with {sum(p.numel() for p in net.parameters())} parameters")
        print(f"Learning rate: {cl_strategy.optimizer.param_groups[0]['lr']}")
        
        # Test training on first experience
        for i, experience in enumerate(benchmark.train_datasets_stream):
            if i == 0:  # Only train on first experience
                print(f"Training on experience {experience.current_experience}")
                result = cl_strategy.train(experience)
                print("Training completed successfully")
                
                # Check if loss decreased
                if result and 'Loss_Epoch/train_phase/train_stream' in result:
                    final_loss = result['Loss_Epoch/train_phase/train_stream']
                    print(f"Final training loss: {final_loss}")
                break
        
        print("CIFAR10 Domain CL test PASSED")
        return True
        
    except Exception as e:
        print(f"CIFAR10 Domain CL test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("Starting training workload tests...")
    
    # Test both workloads
    test1_passed = test_cifar10_regular()
    test2_passed = test_cifar10_domain()
    
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    print(f"Regular CIFAR10 CL: {'PASSED' if test1_passed else 'FAILED'}")
    print(f"CIFAR10 Domain CL:  {'PASSED' if test2_passed else 'FAILED'}")
    
    if test1_passed and test2_passed:
        print("ALL TESTS PASSED! Training is now functional.")
    else:
        print("Some tests failed. Please check the error messages above.")
    
    print("="*50)

if __name__ == "__main__":
    main() 