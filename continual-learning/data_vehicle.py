# File: data_vehicle.py
import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class VehicleYOLODataset(Dataset):
    def __init__(self, root_dir, contexts, scenario='class', transform=None):
        self.root_dir = root_dir
        self.contexts = contexts
        self.scenario = scenario
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        
        # Parse YOLO format dataset
        self.data = []
        self.classes = sorted(os.listdir(os.path.join(root_dir, 'labels')))
        
        # Split into contexts based on scenario
        if scenario == 'class':
            classes_per_context = len(self.classes) // contexts
            self.context_map = {
                i: self.classes[i*classes_per_context:(i+1)*classes_per_context]
                for i in range(contexts)
            }
        # Add other scenario splits as needed
        
        # Load data
        for img_file in os.listdir(os.path.join(root_dir, 'images')):
            label_file = os.path.join(root_dir, 'labels', 
                                    os.path.splitext(img_file)[0] + '.txt')
            with open(label_file, 'r') as f:
                labels = [line.strip().split() for line in f.readlines()]
            
            # Convert YOLO to class labels
            classes = [int(label[0]) for label in labels]
            dominant_class = max(set(classes), key=classes.count)
            
            # Assign to context
            for ctx, ctx_classes in self.context_map.items():
                if dominant_class in ctx_classes:
                    self.data.append({
                        'image': os.path.join(root_dir, 'images', img_file),
                        'class': dominant_class,
                        'context': ctx
                    })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = cv2.imread(item['image'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image)
            
        return {
            'x': image,
            'y': item['class'],
            'context': item['context'],
            'task': item['context']
        }

def get_vehicle_loaders(root_dir, batch_size=32, contexts=5, scenario='class'):
    full_dataset = VehicleYOLODataset(root_dir, contexts=contexts, scenario=scenario)
    
    # Split into train/test
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, full_dataset.context_map
