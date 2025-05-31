#!/usr/bin/env python3
"""
Vehicle dataset implementation for continual learning using the framework from
https://github.com/GMvandeVen/continual-learning
"""

import os
import argparse
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
import pandas as pd
import torch.nn as nn
import glob

# Set random seeds for reproducibility
torch.manual_seed(1)
random.seed(1)
np.random.seed(1)

#################################
## Vehicle dataset implementation
#################################


class VehicleDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, target_size=(640, 640)):
        """
        Custom dataset for vehicle detection with YOLO format annotations
        
        Args:
            root_dir (str): Root directory of the dataset
            split (str): 'train' or 'valid'
            transform (callable, optional): Optional transform to be applied on images
            target_size (tuple): Target size for resizing images (width, height)
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.target_size = target_size
        
        # Define class names and mapping
        self.classes = ['Car', 'Threewheel', 'Bus', 'Truck', 'Motorbike', 'Van']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for idx, cls in enumerate(self.classes)}
        
        # Get image paths from the split directory
        img_path = os.path.join(root_dir, split, 'images')
        self.img_files = sorted(glob.glob(os.path.join(img_path, '*.*')))
        self.img_files = [img for img in self.img_files if img.split('.')[-1].lower() in ['jpg', 'jpeg', 'png', 'bmp']]
        
        # Get corresponding label paths
        self.label_dir = os.path.join(root_dir, split, 'labels')
        self.label_files = [os.path.join(self.label_dir, os.path.basename(img_file).split('.')[0] + '.txt') 
                           for img_file in self.img_files]
        
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        """Get image and target for YOLO format"""
        img_path = self.img_files[idx]
        label_path = self.label_files[idx]
        
        # Load image
        img = Image.open(img_path).convert('RGB')
        orig_width, orig_height = img.size
        
        # Apply transformations to image
        if self.transform:
            img = self.transform(img)
        
        # Read labels (YOLO format: class_id center_x center_y width height)
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    data = line.strip().split()
                    if len(data) == 5:
                        class_id = int(data[0])
                        # YOLO format is center_x, center_y, width, height (normalized)
                        center_x, center_y, width, height = map(float, data[1:])
                        
                        # Store class_id, center_x, center_y, width, height
                        labels.append([class_id, center_x, center_y, width, height])
        
        # Convert to tensor
        target = torch.zeros((len(labels), 5))
        if len(labels) > 0:
            target = torch.tensor(labels)
        
        return img, target, img_path

    def collate_fn(self, batch):
        """Custom collate function for the dataloader to handle variable size targets"""
        imgs, targets, paths = list(zip(*batch))
        
        # Stack images
        imgs = torch.stack([img for img in imgs])
        
        # Return images, targets, and paths
        return imgs, targets, paths
    
    def visualize_item(self, idx):
        """Visualize an image with its bounding boxes for debugging"""
        img, target, path = self[idx]
        
        # Convert tensor to numpy for visualization
        if isinstance(img, torch.Tensor):
            img = img.permute(1, 2, 0).numpy()
            
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        
        # Get image dimensions
        height, width = img.shape[0], img.shape[1]
        
        # Plot each bounding box
        for box in target:
            class_id, x_center, y_center, box_width, box_height = box
            class_name = self.idx_to_class[int(class_id)]
            
            # Convert normalized coordinates to pixel coordinates
            x_min = int((x_center - box_width/2) * width)
            y_min = int((y_center - box_height/2) * height)
            box_width_px = int(box_width * width)
            box_height_px = int(box_height * height)
            
            # Create rectangle patch
            rect = Rectangle((x_min, y_min), box_width_px, box_height_px, 
                            linewidth=2, edgecolor='r', facecolor='none')
            plt.gca().add_patch(rect)
            plt.text(x_min, y_min, class_name, color='white', 
                    bbox=dict(facecolor='red', alpha=0.5))
        
        plt.title(f"Image: {os.path.basename(path)}")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

class VehicleContinualDataset:
    """Class to prepare vehicle dataset for continual learning tasks"""
    
    def __init__(self, root_dir, n_experiences=3, img_size=224, scenario='class'):
        """
        Args:
            root_dir (str): Path to the dataset root
            n_experiences (int): Number of sequential experiences/tasks
            img_size (int): Input image size for the model
            scenario (str): One of 'class', 'domain' (class-incremental or domain-incremental)
        """
        self.root_dir = root_dir
        self.n_experiences = n_experiences
        self.img_size = img_size
        self.scenario = scenario
        self.classes = ['Car', 'Threewheel', 'Bus', 'Truck', 'Motorbike', 'Van']
        
        # Define transformations
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
        ])
        
        # Create full dataset
        self.train_dataset = YOLOVehicleDataset(
            root_dir=root_dir, 
            split='train',
            transform=self.transform
        )
        
        self.val_dataset = YOLOVehicleDataset(
            root_dir=root_dir, 
            split='valid',
            transform=self.transform
        )
        
        # Prepare experiences based on scenario
        self._prepare_experiences()
    
    def _prepare_experiences(self):
        """Prepare dataset division for continual learning"""
        self.train_datasets = []
        self.val_datasets = []
        
        if self.scenario == 'class':
            # Class-incremental: divide classes into different tasks
            classes_per_exp = len(self.classes) // self.n_experiences
            
            for i in range(self.n_experiences):
                start_class = i * classes_per_exp
                end_class = min((i + 1) * classes_per_exp, len(self.classes))
                
                # Get class IDs for this experience
                exp_class_ids = list(range(start_class, end_class))
                
                # Filter train images
                train_indices = [idx for idx, (_, target, _) in enumerate(self.train_dataset) 
                              if any(int(box[0]) in exp_class_ids for box in target)]
                
                # Filter val images
                val_indices = [idx for idx, (_, target, _) in enumerate(self.val_dataset) 
                            if any(int(box[0]) in exp_class_ids for box in target)]
                
                # Create subset datasets
                train_exp_dataset = torch.utils.data.Subset(self.train_dataset, train_indices)
                val_exp_dataset = torch.utils.data.Subset(self.val_dataset, val_indices)
                
                self.train_datasets.append(train_exp_dataset)
                self.val_datasets.append(val_exp_dataset)
                
        elif self.scenario == 'domain':
            # Domain-incremental: randomly divide images (all classes in each task)
            train_indices = list(range(len(self.train_dataset)))
            val_indices = list(range(len(self.val_dataset)))
            
            # Shuffle indices
            random.shuffle(train_indices)
            random.shuffle(val_indices)
            
            # Divide into experiences
            train_per_exp = len(train_indices) // self.n_experiences
            val_per_exp = len(val_indices) // self.n_experiences
            
            for i in range(self.n_experiences):
                start_train = i * train_per_exp
                end_train = (i + 1) * train_per_exp if i < self.n_experiences - 1 else len(train_indices)
                
                start_val = i * val_per_exp
                end_val = (i + 1) * val_per_exp if i < self.n_experiences - 1 else len(val_indices)
                
                # Get indices for this experience
                exp_train_indices = train_indices[start_train:end_train]
                exp_val_indices = val_indices[start_val:end_val]
                
                # Create subset datasets
                train_exp_dataset = torch.utils.data.Subset(self.train_dataset, exp_train_indices)
                val_exp_dataset = torch.utils.data.Subset(self.val_dataset, exp_val_indices)
                
                self.train_datasets.append(train_exp_dataset)
                self.val_datasets.append(val_exp_dataset)
    
    def get_dataloaders(self, batch_size=32, num_workers=4):
        """Get dataloaders for all experiences
        
        Returns:
            train_loaders: List of training dataloaders for each experience
            val_loaders: List of validation dataloaders for each experience
        """
        train_loaders = []
        val_loaders = []
        
        for train_dataset in self.train_datasets:
            train_loader = DataLoader(
                train_dataset, 
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                collate_fn=self.train_dataset.collate_fn,
                pin_memory=True
            )
            train_loaders.append(train_loader)
        
        for val_dataset in self.val_datasets:
            val_loader = DataLoader(
                val_dataset, 
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=self.val_dataset.collate_fn,
                pin_memory=True
            )
            val_loaders.append(val_loader)
        
        return train_loaders, val_loaders


def get_vehicle_datasets(data_dir, n_contexts=3, scenario="class", normalize=True):
    """Load Vehicle dataset and divide it into different contexts for continual learning.

    Args:
        data_dir (str): directory where dataset is stored
        n_contexts (int): number of contexts to create
        scenario (str): training scenario, choices = ["task", "domain", "class"]
        normalize (bool): if True, normalize inputs to ImageNet stats
            
    Returns:
        train_datasets, test_datasets, classes, num_classes
    """
    # Define transforms
    if normalize:
        # Use ImageNet normalization values
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        mean = [0.0, 0.0, 0.0]
        std = [1.0, 1.0, 1.0]
        
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    # Create full datasets
    train_full = VehicleDataset(data_dir, split='train', transform=transform)
    valid_full = VehicleDataset(data_dir, split='valid', transform=transform)
    
    # Get class names and number of classes
    classes = train_full.classes
    n_classes = len(classes)
    
    # Create context-specific datasets based on scenario
    train_datasets = []
    test_datasets = []
    
    if scenario == "class":
        # Split classes into separate contexts
        classes_per_context = n_classes // n_contexts
        for i in range(n_contexts):
            start_class_idx = i * classes_per_context
            end_class_idx = (i + 1) * classes_per_context if i < n_contexts - 1 else n_classes
            context_class_indices = list(range(start_class_idx, end_class_idx))
            
            # Collect indices for this context
            train_indices = [j for j, (_, label) in enumerate(train_full) 
                           if label in context_class_indices]
            test_indices = [j for j, (_, label) in enumerate(valid_full) 
                          if label in context_class_indices]
            
            # Create datasets
            train_set = Subset(train_full, train_indices)
            test_set = Subset(valid_full, test_indices)
            
            train_datasets.append(train_set)
            test_datasets.append(test_set)
            
    elif scenario == "domain":
        # For domain scenario, split dataset randomly into contexts
        train_indices = list(range(len(train_full)))
        test_indices = list(range(len(valid_full)))
        random.shuffle(train_indices)
        random.shuffle(test_indices)
        
        train_indices_per_context = len(train_indices) // n_contexts
        test_indices_per_context = len(test_indices) // n_contexts
        
        for i in range(n_contexts):
            # Train indices for this context
            start_idx = i * train_indices_per_context
            end_idx = (i + 1) * train_indices_per_context if i < n_contexts - 1 else len(train_indices)
            context_train_indices = train_indices[start_idx:end_idx]
            
            # Test indices for this context
            start_idx = i * test_indices_per_context
            end_idx = (i + 1) * test_indices_per_context if i < n_contexts - 1 else len(test_indices)
            context_test_indices = test_indices[start_idx:end_idx]
            
            # Create datasets
            train_set = Subset(train_full, context_train_indices)
            test_set = Subset(valid_full, context_test_indices)
            
            train_datasets.append(train_set)
            test_datasets.append(test_set)
            
    elif scenario == "task":
        # For task scenario, all classes are present but with task-specific labels
        
        # Group indices by class
        train_indices_by_class = {}
        test_indices_by_class = {}
        
        for class_idx in range(n_classes):
            train_indices_by_class[class_idx] = [j for j, (_, label) in enumerate(train_full) 
                                              if label == class_idx]
            test_indices_by_class[class_idx] = [j for j, (_, label) in enumerate(valid_full) 
                                             if label == class_idx]
        
        # Create balanced tasks
        for i in range(n_contexts):
            train_task_indices = []
            test_task_indices = []
            
            # Take a portion of each class for this task
            for class_idx in range(n_classes):
                cls_train_indices = train_indices_by_class[class_idx]
                cls_test_indices = test_indices_by_class[class_idx]
                
                # Split indices into n_contexts parts
                start_idx = (i * len(cls_train_indices)) // n_contexts
                end_idx = ((i + 1) * len(cls_train_indices)) // n_contexts
                train_task_indices.extend(cls_train_indices[start_idx:end_idx])
                
                start_idx = (i * len(cls_test_indices)) // n_contexts
                end_idx = ((i + 1) * len(cls_test_indices)) // n_contexts
                test_task_indices.extend(cls_test_indices[start_idx:end_idx])
            
            # Create datasets
            train_set = Subset(train_full, train_task_indices)
            test_set = Subset(valid_full, test_task_indices)
            
            train_datasets.append(train_set)
            test_datasets.append(test_set)
    
    return train_datasets, test_datasets, classes, n_classes



#################################
## Model definition
#################################

class VehicleModel(nn.Module):
    """Model for vehicle classification with continual learning capabilities"""
    
    def __init__(self, num_classes, scenario="class", contexts=5, fc_units=400, device='cuda'):
        super().__init__()
        
        # Get pre-trained ResNet and remove the last layer
        from torchvision.models import resnet18
        self.feature_extractor = resnet18(pretrained=True)
        for param in self.feature_extractor.parameters():
            param.requires_grad = False  # Freeze the base model
        
        # Get the feature size from the last resnet layer
        feature_size = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Identity()  # Remove the original fc layer
        
        # Define classifier
        if scenario == "task":
            # For task scenario, each task has its own head
            self.classifiers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(feature_size, fc_units),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(fc_units, num_classes)
                ) for _ in range(contexts)
            ])
        else:
            # For domain and class scenarios, single classifier
            self.classifier = nn.Sequential(
                nn.Linear(feature_size, fc_units),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(fc_units, num_classes)
            )
            
        self.scenario = scenario
        self.num_classes = num_classes
        self.device = device
    
    def forward(self, x, context=None):
        """Forward pass through the model"""
        features = self.feature_extractor(x)
        
        if self.scenario == "task" and context is not None:
            return self.classifiers[context](features)
        elif self.scenario == "task":
            # If no context is provided but scenario is task, return logits for all contexts
            outputs = []
            for classifier in self.classifiers:
                outputs.append(classifier(features))
            return torch.cat(outputs, dim=1)
        else:
            return self.classifier(features)

#################################
## Continual Learning methods
#################################

class EWC:
    """Elastic Weight Consolidation implementation"""
    
    def __init__(self, model, device, lambda_=5000):
        self.model = model
        self.device = device
        self.lambda_ = lambda_
        self.params = {n: p for n, p in model.named_parameters() if p.requires_grad}
        self.p_old = {}
        self.F = {}  # Fisher information matrix
        
        # Initialize
        for n, p in self.params.items():
            self.p_old[n] = p.data.clone()
            self.F[n] = torch.zeros_like(p)
    
    def update_fisher(self, dataloader):
        """Compute Fisher information matrix"""
        self.model.eval()
        
        # Store current parameters before updating Fisher
        for n, p in self.params.items():
            self.p_old[n] = p.data.clone()
            self.F[n] = torch.zeros_like(p)
        
        # Compute Fisher information
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.model.zero_grad()
            outputs = self.model(inputs)
            loss = nn.CrossEntropyLoss()(outputs, targets)
            loss.backward()
            
            for n, p in self.params.items():
                if p.grad is not None:
                    self.F[n] += p.grad.data ** 2 / len(dataloader.dataset)
    
    def ewc_loss(self):
        """Compute EWC regularization loss"""
        loss = 0
        for n, p in self.params.items():
            loss += (self.F[n] * (p - self.p_old[n]) ** 2).sum()
        return loss * self.lambda_ / 2

class SynapticIntelligence:
    """Synaptic Intelligence implementation"""
    
    def __init__(self, model, lambda_=0.1):
        self.model = model
        self.lambda_ = lambda_
        self.params = {n: p for n, p in model.named_parameters() if p.requires_grad}
        
        # Initialize importance and checkpoint
        self.W = {}  # Accumulated path integral
        self.importance = {}  # Parameter importance
        self.checkpoint = {}  # Parameter checkpoint for path integral
        
        for n, p in self.params.items():
            self.W[n] = torch.zeros_like(p.data)
            self.importance[n] = torch.zeros_like(p.data)
            self.checkpoint[n] = p.data.clone()
    
    def update_importance(self, train_loader, optimizer):
        """Update importance parameters based on training"""
        # Get current parameters
        for n, p in self.params.items():
            self.checkpoint[n] = p.data.clone()
        
        # Compute importance
        for inputs, targets in train_loader:
            if isinstance(targets, tuple):
                inputs, targets = inputs.to(self.model.device), targets[0].to(self.model.device)
            else:
                inputs, targets = inputs.to(self.model.device), targets.to(self.model.device)
            
            # Forward and backward
            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = nn.CrossEntropyLoss()(outputs, targets)
            loss.backward()
            
            # Accumulate path integral
            for n, p in self.params.items():
                if p.grad is not None:
                    self.W[n] += p.grad.data * (p.data - self.checkpoint[n])
            
            # Update checkpoint
            optimizer.step()
            for n, p in self.params.items():
                self.checkpoint[n] = p.data.clone()
        
        # Update importance for this context
        for n, p in self.params.items():
            delta = p.data - self.checkpoint[n]
            delta_squared = delta ** 2
            eps = 0.1  # Small value to avoid division by zero
            self.importance[n] += self.W[n] / (delta_squared + eps)
            self.W[n] = torch.zeros_like(p.data)  # Reset path integral
    
    def si_loss(self):
        """Compute SI regularization loss"""
        loss = 0
        for n, p in self.params.items():
            loss += (self.importance[n] * (p - self.checkpoint[n]) ** 2).sum()
        return loss * self.lambda_


#################################
## Training and Evaluation
#################################

def train(model, train_loader, optimizer, criterion, device, ewc=None, si=None):
    """Train model for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Add regularization if needed
        if ewc is not None:
            loss += ewc.ewc_loss()
            
        if si is not None:
            loss += si.si_loss()
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total
    
    return train_loss, train_acc

def evaluate(model, test_loader, criterion, device, context=None):
    """Evaluate model performance"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            if context is not None and model.scenario == "task":
                outputs = model(inputs, context=context)
            else:
                outputs = model(inputs)
                
            loss = criterion(outputs, targets)
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    test_loss = running_loss / len(test_loader)
    test_acc = 100 * correct / total
    
    return test_loss, test_acc

def evaluate_all_contexts(model, test_loaders, criterion, device):
    """Evaluate model on all contexts"""
    accuracies = []
    for i, test_loader in enumerate(test_loaders):
        if model.scenario == "task":
            _, acc = evaluate(model, test_loader, criterion, device, context=i)
        else:
            _, acc = evaluate(model, test_loader, criterion, device)
        accuracies.append(acc)
    
    average_acc = sum(accuracies) / len(accuracies)
    return accuracies, average_acc


#################################
## Visualization
#################################

def setup_visdom_plots(vis, title_suffix=""):
    """Setup Visdom visualization plots"""
    from visdom import Visdom
    
    # Loss plot
    loss_win = vis.line(
        X=np.array([0]),
        Y=np.array([0]),
        opts=dict(
            title=f"Training Loss {title_suffix}",
            xlabel='Iterations',
            ylabel='Loss',
            legend=['Train Loss']
        )
    )
    
    # Accuracy plot
    acc_win = vis.line(
        X=np.array([0]),
        Y=np.array([0]),
        opts=dict(
            title=f"Accuracy {title_suffix}",
            xlabel='Epochs',
            ylabel='Accuracy (%)',
            legend=['Overall Accuracy']
        )
    )
    
    # Context-specific accuracy plot
    ctx_acc_win = vis.line(
        X=np.array([0]),
        Y=np.array([0]),
        opts=dict(
            title=f"Context-specific Accuracy {title_suffix}",
            xlabel='Epochs',
            ylabel='Accuracy (%)',
            legend=['Context 1']
        )
    )
    
    return {
        'loss': loss_win,
        'accuracy': acc_win,
        'ctx_accuracy': ctx_acc_win
    }

def update_visdom_plots(vis, vis_wins, iteration, train_loss, epoch, accuracies, avg_accuracy):
    """Update Visdom plots with new data"""
    # Update loss plot
    vis.line(
        X=np.array([iteration]),
        Y=np.array([train_loss]),
        win=vis_wins['loss'],
        update='append'
    )
    
    # Update accuracy plot
    vis.line(
        X=np.array([epoch]),
        Y=np.array([avg_accuracy]),
        win=vis_wins['accuracy'],
        update='append'
    )
    
    # Update context-specific accuracy plot
    for i, acc in enumerate(accuracies):
        vis.line(
            X=np.array([epoch]),
            Y=np.array([acc]),
            win=vis_wins['ctx_accuracy'],
            name=f'Context {i+1}',
            update='append'
        )

def plot_results(results, num_contexts, filename_prefix="vehicle"):
    """Plot and save results from training"""
    # Plot loss
    plt.figure(figsize=(10, 6))
    plt.plot(results['train_loss'], label='Train Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{filename_prefix}_loss.png')
    
    # Plot accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(results['avg_acc'], label='Average Accuracy')
    for i in range(num_contexts):
        plt.plot([acc[i] if i < len(acc) else None for acc in results['accuracies']], 
                label=f'Context {i+1}')
    plt.title('Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{filename_prefix}_accuracy.png')
    
    # Plot forgetting
    max_accs = [0] * num_contexts
    forgetting = [[] for _ in range(num_contexts)]
    
    # Calculate maximum accuracy for each context
    for ep_idx, accs in enumerate(results['accuracies']):
        for ctx_idx, acc in enumerate(accs):
            if ctx_idx < len(accs):
                # Update max accuracy if learning this task
                if ep_idx // results['epochs_per_context'] == ctx_idx:
                    max_accs[ctx_idx] = max(max_accs[ctx_idx], acc)
                
                # Calculate forgetting
                curr_ctx = ep_idx // results['epochs_per_context']
                if curr_ctx > ctx_idx:
                    forgetting[ctx_idx].append(max_accs[ctx_idx] - acc)
                else:
                    forgetting[ctx_idx].append(0)
    
    # Plot forgetting
    plt.figure(figsize=(10, 6))
    for i in range(num_contexts - 1):  # Don't show forgetting for the last context
        plt.plot(forgetting[i], label=f'Context {i+1}')
    plt.title('Forgetting')
    plt.xlabel('Epochs after learning context')
    plt.ylabel('Forgetting (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{filename_prefix}_forgetting.png')


#################################
## Main
#################################

def main():
    """Main function to run the continual learning experiment"""
    parser = argparse.ArgumentParser('Vehicle Continual Learning Experiment')
    
    # Dataset parameters
    parser.add_argument('--data-dir', type=str, default='./data/vehicle-dataset-for-yolo/train',
                        help='directory containing the vehicle dataset')
    
    # Experiment parameters
    parser.add_argument('--scenario', type=str, default='class',
                        choices=['class', 'task', 'domain'],
                        help='continual learning scenario to use')
    parser.add_argument('--contexts', type=int, default=5,
                        help='number of contexts/tasks to split the dataset into')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=20,
                        help='number of epochs to train per context')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    
    # Model parameters
    parser.add_argument('--fc-units', type=int, default=400,
                        help='number of hidden units in FC layers')
    
    # Continual learning parameters
    parser.add_argument('--method', type=str, default='none',
                        choices=['none', 'ewc', 'si'],
                        help='continual learning method to use')
    parser.add_argument('--lambda', type=float, dest='lambda_', default=5000,
                        help='regularization strength for EWC or SI')
    
    # Visualization parameters
    parser.add_argument('--visdom', action='store_true',
                        help='use Visdom for visualization')
    parser.add_argument('--no-plots', action='store_true',
                        help='disable matplotlib plots')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed')
    parser.add_argument('--no-cuda', action='store_true',
                        help='disable CUDA')
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Set device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    print(f"Loading Vehicle dataset from {args.data_dir}...")
    try:
        train_datasets, test_datasets, classes, num_classes = get_vehicle_datasets(
            args.data_dir, args.contexts, args.scenario)
        print(f"Dataset loaded. Found {len(classes)} classes: {classes}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please make sure the dataset is downloaded and has the correct structure:")
        print("- vehicle_annotations.csv: CSV file with image filenames and class labels")
        print("- images/: Directory containing the vehicle images")
        return
    
    # Create model
    print(f"Creating model for {args.scenario} scenario with {args.contexts} contexts...")
    model = VehicleModel(
        num_classes=num_classes if args.scenario != "task" else num_classes // args.contexts,
        scenario=args.scenario,
        contexts=args.contexts,
        fc_units=args.fc_units,
        device=device
    ).to(device)
    
    # Setup optimizer and loss
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], 
        lr=args.lr
    )
    criterion = nn.CrossEntropyLoss()
    
    # Setup continual learning method
    ewc = None
    si = None
    
    if args.method == 'ewc':
        print("Using Elastic Weight Consolidation (EWC)")
        ewc = EWC(model, device, lambda_=args.lambda_)
    elif args.method == 'si':
        print("Using Synaptic Intelligence (SI)")
        si = SynapticIntelligence(model, lambda_=args.lambda_)
    
    # Setup visualization
    vis = None
    vis_wins = None
    
    if args.visdom:
        try:
            from visdom import Visdom
            vis = Visdom()
            if vis.check_connection():
                print("Visdom connected. Visualizations will be shown at http://localhost:8097")
                vis_wins = setup_visdom_plots(vis, f"{args.method}_{args.scenario}")
            else:
                print("Visdom not connected. Run 'python -m visdom.server' in another terminal.")
                args.visdom = False
        except ImportError:
            print("Visdom not installed. Run 'pip install visdom' to use visualizations.")
            args.visdom = False
    
    # Setup results tracking
    results = {
        'train_loss': [],
        'train_acc': [],
        'accuracies': [],
        'avg_acc': [],
        'epochs_per_context': args.epochs
    }
    
    # Create test loaders
    test_loaders = [DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
                   for test_dataset in test_datasets]
    
    # Training loop
    print("\nStarting training...")
    global_iteration = 0
    global_epoch = 0
    
    # Train on each context sequentially
    for context_id, train_dataset in enumerate(train_datasets):
        print(f"\n--- Training on context {context_id+1}/{args.contexts} ---")
        
        # Create data loader
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True
        )
        
        # Train for multiple epochs on current context
        for epoch in range(args.epochs):
            # Training
            train_loss, train_acc = train(
                model, train_loader, optimizer, criterion, device, ewc, si
            )
            
            # Evaluation on all contexts seen so far
            current_test_loaders = test_loaders[:context_id+1]
            ctx_accuracies, avg_accuracy = evaluate_all_contexts(
                model, current_test_loaders, criterion, device
            )
            
            # Pad accuracies to have entries for all contexts
            full_accuracies = ctx_accuracies + [0] * (args.contexts - len(ctx_accuracies))
            
            # Store results
            results['train_loss'].append(train_loss)
            results['train_acc'].append(train_acc)
            results['accuracies'].append(full_accuracies)
            results['avg_acc'].append(avg_accuracy)
            
            # Print progress
            print(f"Epoch {epoch+1}/{args.epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                  f"Avg Test Acc: {avg_accuracy:.2f}%")
            
            for i, acc in enumerate(ctx_accuracies):
                print(f"  Context {i+1}: {acc:.2f}%")
            
            # Update visualizations
            if args.visdom:
                update_visdom_plots(
                    vis, vis_wins, global_iteration, train_loss, 
                    global_epoch, full_accuracies, avg_accuracy
                )
            
            global_iteration += len(train_loader)
            global_epoch += 1
        
        # After training on this context, update CL methods
        if args.method == 'ewc':
            print("Updating EWC Fisher information...")
            ewc.update_fisher(train_loader)
        elif args.method == 'si':
            print("Updating SI importance...")
            si.update_importance(train_loader, optimizer)
    
    # Final evaluation on all contexts
    final_accuracies, final_avg_acc = evaluate_all_contexts(
        model, test_loaders, criterion, device
    )
    
    # Print final results
    print("\n--- Final Results ---")
    print(f"Average Accuracy: {final_avg_acc:.2f}%")
    for i, acc in enumerate(final_accuracies):
        print(f"Context {i+1}: {acc:.2f}%")
    
    # Plot results
    if not args.no_plots:
        print("Plotting results...")
        plot_results(results, args.contexts)
        print("Plots saved as vehicle_loss.png, vehicle_accuracy.png, and vehicle_forgetting.png")
    
    print("\nExperiment completed.")

if __name__ == "__main__":
    main()
