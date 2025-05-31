import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

class VehicleDataset(Dataset):
    """Dataset class for Vehicle Dataset for YOLO converted to classification task."""
    
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir (string): Directory with the dataset.
            split (string): 'train' or 'valid' split.
            transform (callable, optional): Optional transform for images.
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        # Define the classes
        self.classes = ['car', 'motorbike', 'threewheel', 'van', 'bus', 'truck']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Load the annotations and images
        self.images, self.labels = self._load_dataset()
    
    def _load_dataset(self):
        """Load the dataset from YOLO format annotations."""
        images = []
        labels = []
        
        # Path to the annotations directory
        annotations_dir = os.path.join(self.root_dir, self.split, 'labels')
        images_dir = os.path.join(self.root_dir, self.split, 'images')
        
        # Check for directory structure variants
        if not os.path.exists(annotations_dir):
            annotations_dir = os.path.join(self.root_dir, 'labels', self.split)
        
        if not os.path.exists(images_dir):
            images_dir = os.path.join(self.root_dir, 'images', self.split)
        
        # Get all annotation files
        annotation_files = [f for f in os.listdir(annotations_dir) if f.endswith('.txt')]
        
        for ann_file in annotation_files:
            # Check for different image extensions
            image_base = os.path.splitext(ann_file)[0]
            image_path = None
            
            for ext in ['.jpg', '.jpeg', '.png']:
                potential_path = os.path.join(images_dir, image_base + ext)
                if os.path.exists(potential_path):
                    image_path = potential_path
                    break
            
            # Skip if image doesn't exist
            if image_path is None:
                continue
            
            # Read the annotations
            with open(os.path.join(annotations_dir, ann_file), 'r') as f:
                lines = f.readlines()
            
            # Extract class IDs from annotations (YOLO format: class_id x_center y_center width height)
            class_ids = [int(line.strip().split()[0]) for line in lines]
            
            # If there are no annotations, skip this image
            if not class_ids:
                continue
            
            # Use the most frequent class as the label for the whole image
            most_common_class = max(set(class_ids), key=class_ids.count)
            
            images.append(image_path)
            labels.append(most_common_class)
        
        return images, labels
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


import os
import subprocess
import zipfile
from pathlib import Path

def download_and_prepare_vehicle_dataset(data_dir):
    """
    Download and prepare the Vehicle Dataset for YOLO.
    
    Args:
        data_dir (str): Directory where data should be stored.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Path to the downloaded zip file
    zip_path = data_dir / "vehicle-dataset-for-yolo.zip"
    
    # Check if the dataset is already downloaded and extracted
    train_dir = data_dir / "train"
    valid_dir = data_dir / "valid"
    
    if train_dir.exists() and valid_dir.exists():
        print("Vehicle Dataset already exists in the data directory.")
        return
    
    # Try to download using Kaggle API
    try:
        print("Downloading dataset using Kaggle API...")
        subprocess.run(
            ["kaggle", "datasets", "download", "nadinpethiyagoda/vehicle-dataset-for-yolo"],
            cwd=str(data_dir),
            check=True
        )
    except Exception as e:
        print(f"Failed to download using Kaggle API: {e}")
        print("Please download the dataset manually from Kaggle and place it in the data directory.")
        return
    
    # Extract the zip file
    if zip_path.exists():
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(str(data_dir))
        print("Dataset extracted successfully.")
    else:
        print(f"Zip file not found: {zip_path}")


def get_vehicle_datasets(data_dir, n_contexts=6, verbose=False):
    """
    Return the datasets for Vehicle dataset as a list of contexts,
    with each context being a classification task.
    
    Args:
        data_dir (str): Directory where the data is stored or will be downloaded.
        n_contexts (int): Number of contexts to create. Default is 6 (one per vehicle class).
        verbose (bool): If True, print more information.
        
    Returns:
        (tuple): <context_train>, <context_test>
    """
    # Define transformations
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    valid_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Initialize the datasets
    train_dataset = VehicleDataset(data_dir, split='train', transform=train_transform)
    valid_dataset = VehicleDataset(data_dir, split='valid', transform=valid_transform)
    
    if verbose:
        print(f"Full training dataset size: {len(train_dataset)}")
        print(f"Full validation dataset size: {len(valid_dataset)}")
    
    # Get indices for each class
    train_indices_per_class = {class_idx: [] for class_idx in range(len(train_dataset.classes))}
    valid_indices_per_class = {class_idx: [] for class_idx in range(len(valid_dataset.classes))}
    
    for idx, label in enumerate(train_dataset.labels):
        train_indices_per_class[label].append(idx)
    
    for idx, label in enumerate(valid_dataset.labels):
        valid_indices_per_class[label].append(idx)
    
    # Create context datasets
    train_contexts = []
    valid_contexts = []
    
    # Determine classes per context
    if n_contexts == 6:  # One class per context
        classes_per_context = [[i] for i in range(6)]
    elif n_contexts == 3:  # Two classes per context
        classes_per_context = [[0, 1], [2, 3], [4, 5]]
    elif n_contexts == 2:  # Three classes per context
        classes_per_context = [[0, 1, 2], [3, 4, 5]]
    elif n_contexts == 1:  # All classes in one context
        classes_per_context = [[0, 1, 2, 3, 4, 5]]
    else:
        raise ValueError(f"Unsupported number of contexts: {n_contexts}")
    
    # Create the datasets for each context
    for context_classes in classes_per_context:
        # Gather indices for the current context
        train_context_indices = []
        valid_context_indices = []
        
        for class_idx in context_classes:
            train_context_indices.extend(train_indices_per_class[class_idx])
            valid_context_indices.extend(valid_indices_per_class[class_idx])
        
        # Create subset datasets
        train_context_dataset = torch.utils.data.Subset(train_dataset, train_context_indices)
        valid_context_dataset = torch.utils.data.Subset(valid_dataset, valid_context_indices)
        
        train_contexts.append(train_context_dataset)
        valid_contexts.append(valid_context_dataset)
    
    if verbose:
        for i, (train_ds, valid_ds) in enumerate(zip(train_contexts, valid_contexts)):
            print(f"Context {i}: {classes_per_context[i]} - Train size: {len(train_ds)}, Valid size: {len(valid_ds)}")
            class_names = [train_dataset.classes[idx] for idx in classes_per_context[i]]
            print(f"Classes: {class_names}")
    
    return train_contexts, valid_contexts
