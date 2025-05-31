# Import necessary libraries
import os
import glob
import random
import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
import flwr as fl
from flwr.common import NDArrays, Scalar
from typing import Dict, List, Optional, Tuple, Union
from collections import OrderedDict
import time
import copy

#muultiprocessing
import multiprocessing

# Set random seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Define paths and constants
DATA_PATH = "./vehicle-dataset-for-yolo/vehicle dataset/train/"
IMAGES_PATH = os.path.join(DATA_PATH, "images")
LABELS_PATH = os.path.join(DATA_PATH, "labels")
NUM_CLIENTS = 5
BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 416  # YOLO standard input size
NUM_CLASSES = 4  # Assuming 4 classes: car, motorcycle, bus, truck

print(f"Using device: {DEVICE}")

# Custom Dataset class for YOLO format data
class VehicleDataset(Dataset):
    def __init__(self, images_path, labels_path, img_size=416, transforms=None, client_id=None, num_clients=1):
        self.images_path = images_path
        self.labels_path = labels_path
        self.img_size = img_size
        self.transforms = transforms
        
        # Get all image files
        self.img_files = sorted(glob.glob(os.path.join(images_path, "*.jpg")))
        
        # Shard the dataset for federated learning
        if client_id is not None:
            total_data = len(self.img_files)
            data_per_client = total_data // num_clients
            start_idx = client_id * data_per_client
            end_idx = (client_id + 1) * data_per_client if client_id < num_clients - 1 else total_data
            self.img_files = self.img_files[start_idx:end_idx]
            
        print(f"Client {client_id}: Loading {len(self.img_files)} images")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        # Load image
        img_path = self.img_files[index]
        img = Image.open(img_path).convert("RGB")
        
        # Get corresponding label path
        label_path = os.path.join(self.labels_path, os.path.basename(img_path).replace(".jpg", ".txt"))
        
        # Original image dimensions
        width, height = img.size
        
        # Resize image
        if self.transforms:
            img = self.transforms(img)
        
        # Parse YOLO format labels
        boxes = []
        labels = []
        
        if os.path.exists(label_path):
            with open(label_path, "r") as file:
                for line in file:
                    values = line.strip().split()
                    class_id = int(values[0])
                    # YOLO format: class_id, x_center, y_center, width, height (normalized)
                    x_center = float(values[1])
                    y_center = float(values[2])
                    box_width = float(values[3])
                    box_height = float(values[4])
                    
                    # Convert to [x_min, y_min, x_max, y_max] format normalized
                    x_min = x_center - box_width/2
                    y_min = y_center - box_height/2
                    x_max = x_center + box_width/2
                    y_max = y_center + box_height/2
                    
                    # Clip to ensure values are between 0 and 1
                    x_min = max(0, min(1, x_min))
                    y_min = max(0, min(1, y_min))
                    x_max = max(0, min(1, x_max))
                    y_max = max(0, min(1, y_max))
                    
                    boxes.append([x_min, y_min, x_max, y_max])
                    labels.append(class_id)
        
        # Convert to tensor
        if not boxes:
            # No objects in this image, provide dummy box to avoid errors
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            
        # Create target dictionary
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([index])
        }
        
        return img, target

# Define transforms
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Custom model based on YOLOv3 architecture (simplified for this example)
class SimplifiedYOLO(nn.Module):
    def __init__(self, num_classes):
        super(SimplifiedYOLO, self).__init__()
        
        # Use a pre-trained backbone (ResNet)
        backbone = torchvision.models.resnet34(pretrained=True)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        
        # Detection head
        self.head = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, (5 + num_classes) * 3, kernel_size=1)  # 3 anchors per cell, each with 5+num_classes outputs
        )
        
        # Initialize weights
        self._initialize_weights()
        
        # Anchors (simplified)
        self.anchors = torch.tensor([
            [10, 13], [16, 30], [33, 23],
            [30, 61], [62, 45], [59, 119],
            [116, 90], [156, 198], [373, 326]
        ], dtype=torch.float32).view(3, 3, 2)  # 3 scales, 3 anchors per scale
        
    def _initialize_weights(self):
        for m in self.head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
                
    def forward(self, x):
        # Feature extraction
        features = self.backbone(x)
        
        # Detection head
        output = self.head(features)
        
        # Reshape: [batch, (5+num_classes)*3, grid_h, grid_w] -> [batch, 3, grid_h, grid_w, 5+num_classes]
        batch_size, _, grid_h, grid_w = output.shape
        output = output.view(batch_size, 3, -1, grid_h, grid_w).permute(0, 1, 3, 4, 2)
        
        return output

# Custom loss function for YOLO
class YOLOLoss(nn.Module):
    def __init__(self, num_classes=4):
        super(YOLOLoss, self).__init__()
        self.num_classes = num_classes
        self.mse_loss = nn.MSELoss(reduction="sum")
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="sum")
        self.lambda_coord = 5.0
        self.lambda_noobj = 0.5
        
    def forward(self, predictions, targets):
        # Simplified YOLO loss calculation
        # This is a placeholder for the actual YOLO loss function
        # In a real implementation, this would be much more complex
        
        # Separate the predictions
        batch_size = predictions.shape[0]
        total_loss = torch.zeros(1, device=DEVICE)
        
        for i in range(batch_size):
            # Extract target boxes and labels
            target_boxes = targets[i]["boxes"]  # [num_obj, 4] - normalized [x_min, y_min, x_max, y_max]
            target_labels = targets[i]["labels"]  # [num_obj]
            
            if len(target_boxes) == 0:
                continue
                
            # Convert predictions to bounding boxes
            # This is simplified and would be more complex in a real implementation
            pred = predictions[i]  # [3, grid_h, grid_w, 5+num_classes]
            
            # Calculate losses (placeholder calculations)
            # In reality, this would involve:
            # 1. Converting predictions to the same format as targets
            # 2. Calculating IoU to assign targets to predictions
            # 3. Computing coordinate, objectness, and class losses
            
            # Dummy loss calculation
            obj_mask = torch.ones(1, device=DEVICE)  # Which cells have objects
            coord_loss = self.lambda_coord * self.mse_loss(pred[0, 0, 0, :4], target_boxes[0])  # Coordinate loss
            obj_loss = self.bce_loss(pred[0, 0, 0, 4], obj_mask)  # Objectness loss
            
            # One-hot encode class targets
            class_target = torch.zeros(self.num_classes, device=DEVICE)
            if len(target_labels) > 0:
                class_target[target_labels[0]] = 1
                
            class_loss = self.bce_loss(pred[0, 0, 0, 5:], class_target)  # Class loss
            
            # Total loss
            total_loss += coord_loss + obj_loss + class_loss
            
        return total_loss / batch_size

# Custom metrics for evaluation
class DetectionMetrics:
    def __init__(self, iou_threshold=0.5):
        self.iou_threshold = iou_threshold
        self.reset()
        
    def reset(self):
        self.correct = 0
        self.total = 0
        self.precision = 0
        self.recall = 0
        self.f1 = 0
        
    def update(self, predictions, targets):
        """Update metrics based on predictions and targets"""
        batch_size = len(predictions)
        
        for i in range(batch_size):
            pred_boxes = predictions[i]["boxes"]
            pred_labels = predictions[i]["labels"]
            pred_scores = predictions[i]["scores"]
            
            target_boxes = targets[i]["boxes"]
            target_labels = targets[i]["labels"]
            
            # Calculate IoU between predictions and targets
            if len(pred_boxes) > 0 and len(target_boxes) > 0:
                ious = self._calculate_iou(pred_boxes, target_boxes)
                
                # Count true positives, false positives, false negatives
                for pred_idx, pred_label in enumerate(pred_labels):
                    # Find best matching target
                    max_iou_idx = ious[pred_idx].argmax().item()
                    max_iou = ious[pred_idx, max_iou_idx].item()
                    
                    if max_iou >= self.iou_threshold and pred_label == target_labels[max_iou_idx]:
                        self.correct += 1
                        
                self.total += len(target_boxes)
                
                # Calculate precision and recall
                if len(pred_boxes) > 0:
                    self.precision = self.correct / len(pred_boxes)
                    
                if len(target_boxes) > 0:
                    self.recall = self.correct / len(target_boxes)
                    
                # Calculate F1 score
                if self.precision + self.recall > 0:
                    self.f1 = 2 * self.precision * self.recall / (self.precision + self.recall)
    
    def _calculate_iou(self, boxes1, boxes2):
        """Calculate IoU between two sets of boxes"""
        # Convert boxes to 2D tensors if they are not already
        boxes1 = boxes1.reshape(-1, 4)
        boxes2 = boxes2.reshape(-1, 4)
        
        # Calculate intersection areas
        x1 = torch.max(boxes1[:, None, 0], boxes2[:, 0])  # [N,M]
        y1 = torch.max(boxes1[:, None, 1], boxes2[:, 1])  # [N,M]
        x2 = torch.min(boxes1[:, None, 2], boxes2[:, 2])  # [N,M]
        y2 = torch.min(boxes1[:, None, 3], boxes2[:, 3])  # [N,M]
        
        # Clip to ensure w, h >= 0
        w = torch.clamp(x2 - x1, min=0)  # [N,M]
        h = torch.clamp(y2 - y1, min=0)  # [N,M]
        
        inter = w * h  # [N,M]
        
        # Calculate areas
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])  # [N]
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])  # [M]
        
        # Calculate union
        union = area1[:, None] + area2 - inter  # [N,M]
        
        # Calculate IoU
        iou = inter / union  # [N,M]
        
        return iou
    
    def get_metrics(self):
        """Return the current metrics"""
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1
        }

# Function to convert model outputs to bounding boxes
def convert_predictions_to_boxes(predictions, conf_threshold=0.5):
    """Convert YOLO predictions to bounding boxes"""
    # This is a simplified version of the conversion
    # In a real implementation, this would involve:
    # 1. Applying sigmoid/exponential to appropriate parts of the prediction
    # 2. Converting from grid coordinates to image coordinates
    # 3. Applying anchor boxes
    # 4. Non-maximum suppression
    
    batch_size, num_anchors, grid_h, grid_w, box_attr = predictions.shape
    result = []
    
    for i in range(batch_size):
        pred = predictions[i]  # [num_anchors, grid_h, grid_w, box_attr]
        
        # Extract components
        box_centers = torch.sigmoid(pred[..., :2])  # Center x, y
        box_sizes = torch.exp(pred[..., 2:4])  # Width, height
        objectness = torch.sigmoid(pred[..., 4])  # Objectness score
        class_probs = torch.softmax(pred[..., 5:], dim=-1)  # Class probabilities
        
        # Create grid
        grid_y, grid_x = torch.meshgrid(torch.arange(grid_h, device=DEVICE), 
                                         torch.arange(grid_w, device=DEVICE),
                                         indexing='ij')
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).repeat(num_anchors, 1, 1, 1)
        
        # Add grid offset and scale by grid size
        box_centers = (box_centers + grid) / torch.tensor([grid_w, grid_h], device=DEVICE)
        box_sizes = box_sizes / torch.tensor([grid_w, grid_h], device=DEVICE)
        
        # Convert to [x_min, y_min, x_max, y_max] format
        x_min = box_centers[..., 0] - box_sizes[..., 0] / 2
        y_min = box_centers[..., 1] - box_sizes[..., 1] / 2
        x_max = box_centers[..., 0] + box_sizes[..., 0] / 2
        y_max = box_centers[..., 1] + box_sizes[..., 1] / 2
        
        # Stack boxes
        boxes = torch.stack([x_min, y_min, x_max, y_max], dim=-1)
        
        # Reshape
        boxes = boxes.reshape(-1, 4)
        objectness = objectness.reshape(-1)
        class_probs = class_probs.reshape(-1, class_probs.shape[-1])
        
        # Filter by confidence
        mask = objectness > conf_threshold
        boxes = boxes[mask]
        scores = objectness[mask]
        class_probs = class_probs[mask]
        
        if len(boxes) > 0:
            # Get class with highest probability
            class_scores, class_ids = class_probs.max(dim=1)
            scores = scores * class_scores
            
            # Create dictionary for this batch item
            result.append({
                "boxes": boxes,
                "labels": class_ids,
                "scores": scores
            })
        else:
            result.append({
                "boxes": torch.zeros((0, 4), device=DEVICE),
                "labels": torch.zeros(0, dtype=torch.int64, device=DEVICE),
                "scores": torch.zeros(0, device=DEVICE)
            })
            
    return result

# Training function
def train(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    
    for batch_idx, (images, targets) in enumerate(train_loader):
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        optimizer.zero_grad()
        predictions = model(images)
        loss = criterion(predictions, targets)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
    
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}")
    return avg_loss

# Validation function
def validate(model, val_loader, criterion, device, metrics):
    model.eval()
    running_loss = 0.0
    metrics.reset()
    
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            predictions = model(images)
            loss = criterion(predictions, targets)
            running_loss += loss.item()
            
            # Convert model outputs to bounding boxes
            box_predictions = convert_predictions_to_boxes(predictions)
            
            # Update metrics
            metrics.update(box_predictions, targets)
    
    avg_loss = running_loss / len(val_loader)
    metrics_dict = metrics.get_metrics()
    
    print(f"Validation Loss: {avg_loss:.4f}")
    print(f"Precision: {metrics_dict['precision']:.4f}, Recall: {metrics_dict['recall']:.4f}, F1: {metrics_dict['f1']:.4f}")
    
    return avg_loss, metrics_dict

# Visualization function
def visualize_predictions(model, data_loader, device, num_images=5):
    model.eval()
    images_so_far = 0
    fig = plt.figure(figsize=(15, 15))
    
    class_names = ["car", "motorcycle", "bus", "truck"]  # Adjust based on actual classes
    
    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(device)
            
            # Get predictions
            predictions = model(images)
            box_predictions = convert_predictions_to_boxes(predictions)
            
            batch_size = images.size(0)
            for i in range(batch_size):
                if images_so_far >= num_images:
                    return fig
                
                images_so_far += 1
                ax = fig.add_subplot(math.ceil(num_images / 2), 2, images_so_far)
                
                # Denormalize image
                img = images[i].cpu().permute(1, 2, 0).numpy()
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img = std * img + mean
                img = np.clip(img, 0, 1)
                
                ax.imshow(img)
                
                # Draw ground truth boxes (green)
                for box, label in zip(targets[i]["boxes"].cpu().numpy(), targets[i]["labels"].cpu().numpy()):
                    x_min, y_min, x_max, y_max = box
                    rect = plt.Rectangle((x_min * IMG_SIZE, y_min * IMG_SIZE),
                                         (x_max - x_min) * IMG_SIZE,
                                         (y_max - y_min) * IMG_SIZE,
                                         fill=False, edgecolor='green', linewidth=2)
                    ax.add_patch(rect)
                    ax.text(x_min * IMG_SIZE, y_min * IMG_SIZE - 5,
                            f"GT: {class_names[label]}",
                            color='green', fontsize=10)
                
                # Draw predicted boxes (red)
                if len(box_predictions[i]["boxes"]) > 0:
                    for box, label, score in zip(box_predictions[i]["boxes"].cpu().numpy(),
                                              box_predictions[i]["labels"].cpu().numpy(),
                                              box_predictions[i]["scores"].cpu().numpy()):
                        x_min, y_min, x_max, y_max = box
                        rect = plt.Rectangle((x_min * IMG_SIZE, y_min * IMG_SIZE),
                                             (x_max - x_min) * IMG_SIZE,
                                             (y_max - y_min) * IMG_SIZE,
                                             fill=False, edgecolor='red', linewidth=2)
                        ax.add_patch(rect)
                        ax.text(x_min * IMG_SIZE, (y_min * IMG_SIZE) - 15,
                                f"Pred: {class_names[label]} ({score:.2f})",
                                color='red', fontsize=10)
                
                ax.set_title(f"Image {images_so_far}")
                ax.axis('off')
                
    return fig

# Plot training progress
def plot_training_progress(train_losses, val_losses, metrics_history):
    # Plot losses
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    # Plot precision and recall
    plt.subplot(1, 3, 2)
    precisions = [m['precision'] for m in metrics_history]
    recalls = [m['recall'] for m in metrics_history]
    plt.plot(precisions, label='Precision')
    plt.plot(recalls, label='Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.legend()
    plt.title('Precision and Recall')
    
    # Plot F1 score
    plt.subplot(1, 3, 3)
    f1_scores = [m['f1'] for m in metrics_history]
    plt.plot(f1_scores, label='F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.title('F1 Score')
    
    plt.tight_layout()
    plt.savefig('training_progress.png')
    plt.close()

# Function to prepare datasets for each client
def prepare_datasets_for_client(client_id):
    train_dataset = VehicleDataset(
        images_path=IMAGES_PATH,
        labels_path=LABELS_PATH,
        transforms=transform,
        client_id=client_id,
        num_clients=NUM_CLIENTS
    )
    
    # Split into train and validation sets (80/20)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    def collate_fn(batch):
        images, targets = zip(*batch)
        return torch.stack(images, 0), list(targets)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, collate_fn=collate_fn)
    
    return train_loader, val_loader

# Flower client implementation for federated learning
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, model, train_loader, val_loader, epochs):
        self.cid = cid
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.criterion = YOLOLoss(num_classes=NUM_CLASSES)
        self.optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        self.metrics = DetectionMetrics()
        self.train_losses = []
        self.val_losses = []
        self.metrics_history = []
        
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        
        # Training loop
        for epoch in range(self.epochs):
            print(f"Client {self.cid}, Epoch {epoch+1}/{self.epochs}")
            train_loss = train(self.model, self.train_loader, self.criterion, self.optimizer, DEVICE, epoch+1)
            val_loss, metrics = validate(self.model, self.val_loader, self.criterion, DEVICE, self.metrics)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.metrics_history.append(metrics)
        
        # Plot training progress for this client
        plot_training_progress(self.train_losses, self.val_losses, self.metrics_history)
        
        # Visualize some predictions
        if self.cid == 0:  # Only for the first client to avoid too many plots
            fig = visualize_predictions(self.model, self.val_loader, DEVICE)
            plt.savefig(f'client_{self.cid}_predictions.png')
            plt.close(fig)
        
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}
    
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        
        # Validation
        loss, metrics = validate(self.model, self.val_loader, self.criterion, DEVICE, self.metrics)
        
        return float(loss), len(self.val_loader.dataset), metrics

# Function to start a Flower client
def start_client(cid):
    # Set up model
    model = SimplifiedYOLO(num_classes=NUM_CLASSES).to(DEVICE)
    
    # Prepare datasets
    train_loader, val_loader = prepare_datasets_for_client(cid)
    
    # Create client
    client = FlowerClient(cid, model, train_loader, val_loader, epochs=NUM_EPOCHS)
    
    # Start client
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)

# Federated learning strategy with continual learning
class ContinualFederatedStrategy(fl.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.round_metrics = []
        
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, float, Dict[str, Scalar]]],
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results from multiple clients."""
        if not results:
            return None, {}
            
        # Aggregate loss weighted by number of examples
        loss_aggregated, metrics_aggregated = super().aggregate_evaluate(server_round, results, failures)
        
        # Store metrics for visualization
        round_metrics = {
            "round": server_round,
            "loss": loss_aggregated,
            **metrics_aggregated
        }
        self.round_metrics.append(round_metrics)
        
        # Save metrics to file
        np.save("federated_metrics.npy", self.round_metrics)
        
        # Print aggregated metrics
        print(f"Round {server_round} completed:")
        print(f"Aggregated loss: {loss_aggregated:.4f}")
        for metric_name, metric_value in metrics_aggregated.items():
            print(f"Aggregated {metric_name}: {metric_value:.4f}")
            
        return loss_aggregated, metrics_aggregated
        
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[fl.common.Parameters], Dict[str, Scalar]]:
        """Aggregate training results from multiple clients with continual learning."""
        # Implement continual learning logic here
# Standard FedAvg aggregation
        aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)
        
        # For continual learning, we could implement:
        # 1. Knowledge distillation
        # 2. Elastic weight consolidation
        # 3. Experience replay
        # Here we'll use a simplified approach by maintaining a global model
        
        # Save global model parameters after each round for continual learning
        if aggregated_parameters is not None:
            # Convert parameters to numpy arrays
            parameters = fl.common.parameters_to_ndarrays(aggregated_parameters)
            
            # Create a model and load parameters
            global_model = SimplifiedYOLO(num_classes=NUM_CLASSES)
            params_dict = zip(global_model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            global_model.load_state_dict(state_dict, strict=True)
            
            # Save model
            torch.save({
                'round': server_round,
                'model_state_dict': global_model.state_dict(),
                'metrics': metrics
            }, f'global_model_round_{server_round}.pth')
            
            print(f"Saved global model for round {server_round}")
            
        return aggregated_parameters, metrics

# Function to visualize federated learning results
def visualize_federated_results(metrics_file="federated_metrics.npy"):
    if not os.path.exists(metrics_file):
        print(f"Metrics file {metrics_file} not found")
        return
        
    metrics = np.load(metrics_file, allow_pickle=True)
    
    # Convert to list of dictionaries if it's not already
    if not isinstance(metrics, list):
        metrics = metrics.tolist()
    
    # Extract data
    rounds = [m['round'] for m in metrics]
    losses = [m['loss'] for m in metrics]
    precisions = [m.get('precision', 0) for m in metrics]
    recalls = [m.get('recall', 0) for m in metrics]
    f1_scores = [m.get('f1', 0) for m in metrics]
    
    # Create visualization
    plt.figure(figsize=(20, 10))
    
    # Plot loss
    plt.subplot(2, 2, 1)
    plt.plot(rounds, losses, 'o-', label='Loss')
    plt.xlabel('Round')
    plt.ylabel('Loss')
    plt.title('Federated Training Loss')
    plt.grid(True)
    
    # Plot precision
    plt.subplot(2, 2, 2)
    plt.plot(rounds, precisions, 'o-', label='Precision')
    plt.xlabel('Round')
    plt.ylabel('Precision')
    plt.title('Federated Precision')
    plt.grid(True)
    
    # Plot recall
    plt.subplot(2, 2, 3)
    plt.plot(rounds, recalls, 'o-', label='Recall')
    plt.xlabel('Round')
    plt.ylabel('Recall')
    plt.title('Federated Recall')
    plt.grid(True)
    
    # Plot F1 score
    plt.subplot(2, 2, 4)
    plt.plot(rounds, f1_scores, 'o-', label='F1 Score')
    plt.xlabel('Round')
    plt.ylabel('F1 Score')
    plt.title('Federated F1 Score')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('federated_learning_results.png')
    plt.close()

# Function to visualize catastrophic forgetting effects in continual learning
def visualize_catastrophic_forgetting(model_checkpoints, data_loader):
    """Visualize how model performance changes across rounds on the same dataset"""
    if not model_checkpoints:
        print("No model checkpoints provided")
        return
        
    # Load models
    models = []
    rounds = []
    
    for checkpoint_path in model_checkpoints:
        checkpoint = torch.load(checkpoint_path)
        model = SimplifiedYOLO(num_classes=NUM_CLASSES).to(DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        models.append(model)
        rounds.append(checkpoint['round'])
    
    # Evaluate each model on the same dataset
    metrics_list = []
    
    for i, model in enumerate(models):
        metrics = DetectionMetrics()
        criterion = YOLOLoss(num_classes=NUM_CLASSES)
        
        loss, model_metrics = validate(model, data_loader, criterion, DEVICE, metrics)
        
        metrics_list.append({
            'round': rounds[i],
            'loss': loss,
            **model_metrics
        })
    
    # Visualize metrics across rounds
    plt.figure(figsize=(15, 10))
    
    # Sort by round
    metrics_list.sort(key=lambda x: x['round'])
    
    # Extract data
    checkpoint_rounds = [m['round'] for m in metrics_list]
    losses = [m['loss'] for m in metrics_list]
    precisions = [m['precision'] for m in metrics_list]
    recalls = [m['recall'] for m in metrics_list]
    f1_scores = [m['f1'] for m in metrics_list]
    
    # Plot all metrics
    plt.subplot(2, 2, 1)
    plt.plot(checkpoint_rounds, losses, 'o-', label='Loss')
    plt.xlabel('Round')
    plt.ylabel('Loss')
    plt.title('Loss Across Rounds (Same Dataset)')
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(checkpoint_rounds, precisions, 'o-', label='Precision')
    plt.xlabel('Round')
    plt.ylabel('Precision')
    plt.title('Precision Across Rounds (Same Dataset)')
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.plot(checkpoint_rounds, recalls, 'o-', label='Recall')
    plt.xlabel('Round')
    plt.ylabel('Recall')
    plt.title('Recall Across Rounds (Same Dataset)')
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.plot(checkpoint_rounds, f1_scores, 'o-', label='F1 Score')
    plt.xlabel('Round')
    plt.ylabel('F1 Score')
    plt.title('F1 Score Across Rounds (Same Dataset)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('catastrophic_forgetting_analysis.png')
    plt.close()
    
    return metrics_list

# Main function to run the federated learning server
def main():
    # Define the strategy
    strategy = ContinualFederatedStrategy(
        fraction_fit=1.0,  # Sample 100% of available clients for training
        fraction_evaluate=1.0,  # Sample 100% of available clients for evaluation
        min_fit_clients=2,  # Minimum number of clients for training
        min_evaluate_clients=2,  # Minimum number of clients for evaluation
        min_available_clients=2,  # Minimum number of available clients
    )
    
    # Start server
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
    )
    
    # After federated learning completes, visualize results
    visualize_federated_results()
    
    # Analyze catastrophic forgetting
    model_checkpoints = glob.glob("global_model_round_*.pth")
    
    if model_checkpoints:
        # Create a test dataset for catastrophic forgetting analysis
        test_dataset = VehicleDataset(
            images_path=IMAGES_PATH,
            labels_path=LABELS_PATH,
            transforms=transform
        )
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
        
        visualize_catastrophic_forgetting(model_checkpoints, test_loader)

# Function to demonstrate continual learning by sequentially training on different tasks
def run_continual_learning_experiment():
    # For this experiment, we'll split the dataset into "tasks"
    # Each task represents a different subset of classes or domains
    
    # Create the base model
    model = SimplifiedYOLO(num_classes=NUM_CLASSES).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = YOLOLoss(num_classes=NUM_CLASSES)
    
    # Task definitions (for example, by class)
    tasks = [
        {"name": "cars", "classes": [0]},
        {"name": "motorcycles", "classes": [1]},
        {"name": "buses_trucks", "classes": [2, 3]}
    ]
    
    all_task_metrics = []
    
    # Function to filter dataset by class
    def filter_dataset_by_classes(dataset, classes):
        filtered_indices = []
        for i in range(len(dataset)):
            _, target = dataset[i]
            # Check if any of the labels are in the specified classes
            if any(label.item() in classes for label in target["labels"]):
                filtered_indices.append(i)
        return torch.utils.data.Subset(dataset, filtered_indices)
    
    # Create the full dataset
    full_dataset = VehicleDataset(
        images_path=IMAGES_PATH,
        labels_path=LABELS_PATH,
        transforms=transform
    )
    
    # Split into train and test
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    
    # Create test loader for evaluation
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # Train on each task sequentially
    for task_idx, task in enumerate(tasks):
        print(f"\n--- Training on Task {task_idx+1}: {task['name']} ---")
        
        # Filter dataset for current task
        task_train_dataset = filter_dataset_by_classes(train_dataset, task["classes"])
        task_train_loader = DataLoader(task_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
        
        # Train on the current task
        task_train_losses = []
        task_metrics_history = []
        
        for epoch in range(NUM_EPOCHS):
            print(f"Task {task_idx+1}, Epoch {epoch+1}/{NUM_EPOCHS}")
            train_loss = train(model, task_train_loader, criterion, optimizer, DEVICE, epoch+1)
            
            # Evaluate on the full test set to measure catastrophic forgetting
            metrics = DetectionMetrics()
            val_loss, metrics_dict = validate(model, test_loader, criterion, DEVICE, metrics)
            
            task_train_losses.append(train_loss)
            task_metrics_history.append(metrics_dict)
            
            print(f"Task {task_idx+1}, Epoch {epoch+1}, Full Test Loss: {val_loss:.4f}")
            print(f"Precision: {metrics_dict['precision']:.4f}, Recall: {metrics_dict['recall']:.4f}, F1: {metrics_dict['f1']:.4f}")
        
        # Save model after each task
        torch.save({
            'task': task_idx,
            'task_name': task['name'],
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': task_train_losses,
            'metrics_history': task_metrics_history
        }, f'continual_learning_task_{task_idx+1}.pth')
        
        all_task_metrics.append({
            'task_idx': task_idx,
            'task_name': task['name'],
            'train_losses': task_train_losses,
            'metrics_history': task_metrics_history
        })
        
        # Visualize some predictions after this task
        fig = visualize_predictions(model, test_loader, DEVICE)
        plt.savefig(f'task_{task_idx+1}_predictions.png')
        plt.close(fig)
    
    # Visualize continual learning results
    visualize_continual_learning_results(all_task_metrics)

# Function to visualize continual learning results
def visualize_continual_learning_results(all_task_metrics):
    if not all_task_metrics:
        print("No task metrics provided")
        return
    
    plt.figure(figsize=(20, 15))
    
    # Plot training loss for each task
    plt.subplot(2, 2, 1)
    for task_metrics in all_task_metrics:
        task_idx = task_metrics['task_idx']
        task_name = task_metrics['task_name']
        train_losses = task_metrics['train_losses']
        epochs = range(1, len(train_losses) + 1)
        plt.plot(epochs, train_losses, 'o-', label=f"Task {task_idx+1}: {task_name}")
    
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss Across Tasks')
    plt.legend()
    plt.grid(True)
    
    # Plot F1 score progression
    plt.subplot(2, 2, 2)
    for task_metrics in all_task_metrics:
        task_idx = task_metrics['task_idx']
        task_name = task_metrics['task_name']
        metrics_history = task_metrics['metrics_history']
        f1_scores = [m['f1'] for m in metrics_history]
        epochs = range(1, len(f1_scores) + 1)
        plt.plot(epochs, f1_scores, 'o-', label=f"Task {task_idx+1}: {task_name}")
    
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('F1 Score Progression Across Tasks')
    plt.legend()
    plt.grid(True)
    
    # Plot precision progression
    plt.subplot(2, 2, 3)
    for task_metrics in all_task_metrics:
        task_idx = task_metrics['task_idx']
        task_name = task_metrics['task_name']
        metrics_history = task_metrics['metrics_history']
        precisions = [m['precision'] for m in metrics_history]
        epochs = range(1, len(precisions) + 1)
        plt.plot(epochs, precisions, 'o-', label=f"Task {task_idx+1}: {task_name}")
    
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.title('Precision Progression Across Tasks')
    plt.legend()
    plt.grid(True)
    
    # Plot recall progression
    plt.subplot(2, 2, 4)
    for task_metrics in all_task_metrics:
        task_idx = task_metrics['task_idx']
        task_name = task_metrics['task_name']
        metrics_history = task_metrics['metrics_history']
        recalls = [m['recall'] for m in metrics_history]
        epochs = range(1, len(recalls) + 1)
        plt.plot(epochs, recalls, 'o-', label=f"Task {task_idx+1}: {task_name}")
    
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.title('Recall Progression Across Tasks')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('continual_learning_results.png')
    plt.close()

# Launch multiple clients in separate processes
def launch_clients(num_clients):
    processes = []
    for i in range(num_clients):
        p = multiprocessing.Process(target=start_client, args=(i,))
        p.start()
        processes.append(p)
    
    return processes

if __name__ == "__main__":
    # Choose the experiment to run
    experiment = "federated_continual"  # Options: "federated", "continual", "federated_continual"
    
    if experiment == "federated":
        # Launch federated learning clients
        client_processes = launch_clients(NUM_CLIENTS)
        
        # Start the server
        main()
        
        # Wait for clients to finish
        for p in client_processes:
            p.join()
            
    elif experiment == "continual":
        # Run continual learning experiment (sequential tasks)
        run_continual_learning_experiment()
        
    elif experiment == "federated_continual":
        # Launch federated learning clients
        client_processes = launch_clients(NUM_CLIENTS)
        
        # Start the server with continual learning strategy
        main()
        
        # Wait for clients to finish
        for p in client_processes:
            p.join()
            
        # Run additional continual learning analysis
        model_checkpoints = sorted(glob.glob("global_model_round_*.pth"))
        if model_checkpoints:
            test_dataset = VehicleDataset(
                images_path=IMAGES_PATH,
                labels_path=LABELS_PATH,
                transforms=transform
            )
            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
            visualize_catastrophic_forgetting(model_checkpoints, test_loader)
    
    print("Experiment completed!")