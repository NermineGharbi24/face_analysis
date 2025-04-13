# train.py
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from face_analysis import FaceAnalysisModel
import cv2
import random

class FairFaceDataset(Dataset):
    """
    Dataset class for the FairFace dataset.
    """
    def __init__(self, csv_file, img_dir, transform=None):
        """
        Args:
            csv_file: Path to the CSV file with annotations
            img_dir: Directory with the images
            transform: Optional transform to be applied
        """
        self.data_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        
        # Map labels to indices
        self.gender_map = {'Male': 0, 'Female': 1}
        self.age_map = {
            '0-2': 0, '3-9': 1, '10-19': 2, '20-29': 3, 
            '30-39': 4, '40-49': 5, '50-59': 6, '60-69': 7, '70+': 8,
            'more than 70': 8
        }
        self.race_map = {
            'White': 0, 'Black': 1, 'East Asian': 2, 'Southeast Asian': 3,
            'Indian': 4, 'Middle Eastern': 5, 'Latino_Hispanic': 6
        }
        
        # Handle possible different column names
        if 'file' in self.data_frame.columns and 'file_name' not in self.data_frame.columns:
            self.data_frame.rename(columns={'file': 'file_name'}, inplace=True)
        
        # Check if we need to process the DataFrame for missing columns
        self._preprocess_dataframe()
        
        # Cleanup any missing files
        self._cleanup_missing_files()
        
    def _preprocess_dataframe(self):
        """Process the DataFrame to ensure all required columns are present."""
        # Handle age column variations
        if 'age' not in self.data_frame.columns and 'age_group' in self.data_frame.columns:
            self.data_frame.rename(columns={'age_group': 'age'}, inplace=True)
            
        # Handle race column variations
        if 'race' not in self.data_frame.columns:
            # Check if we have race score columns
            race_score_cols = [col for col in self.data_frame.columns if col.startswith('race_score')]
            if race_score_cols:
                race_names = ['White', 'Black', 'East Asian', 'Southeast Asian', 'Indian', 'Middle Eastern', 'Latino_Hispanic']
                # Get the race with the highest score
                race_indices = self.data_frame[race_score_cols].values.argmax(axis=1)
                self.data_frame['race'] = [race_names[idx] for idx in race_indices]
                
        # Handle gender column variations
        if 'gender' not in self.data_frame.columns and 'sex' in self.data_frame.columns:
            self.data_frame.rename(columns={'sex': 'gender'}, inplace=True)
            
    def _cleanup_missing_files(self):
        """Remove entries for images that don't exist."""
        valid_indices = []
        
        for idx, row in tqdm(self.data_frame.iterrows(), desc="Checking files", total=len(self.data_frame)):
            file_name = row['file_name']
            img_path = os.path.join(self.img_dir, file_name)
            
            if os.path.exists(img_path):
                valid_indices.append(idx)
        
        # Filter DataFrame to keep only valid entries
        print(f"Found {len(valid_indices)} valid files out of {len(self.data_frame)}")
        self.data_frame = self.data_frame.loc[valid_indices].reset_index(drop=True)
        
    def __len__(self):
        return len(self.data_frame)
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_name = os.path.join(self.img_dir, self.data_frame.iloc[idx]['file_name'])
        
        try:
            image = Image.open(img_name).convert('RGB')
        except (IOError, OSError):
            # If there's an error reading the image, return a random valid image
            random_idx = random.randint(0, len(self.data_frame) - 1)
            while random_idx == idx:
                random_idx = random.randint(0, len(self.data_frame) - 1)
            return self.__getitem__(random_idx)
        
        gender = self.gender_map[self.data_frame.iloc[idx]['gender']]
        age = self.age_map[self.data_frame.iloc[idx]['age']]
        race = self.race_map[self.data_frame.iloc[idx]['race']]
        
        if self.transform:
            image = self.transform(image)
            
        return image, gender, age, race

def train_model(args):
    """
    Train the face analysis model.
    
    Args:
        args: Command line arguments
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }
    
    # Create datasets
    print("Loading training dataset...")
    train_dataset = FairFaceDataset(
        csv_file=args.csv_file,
        img_dir=args.img_dir,
        transform=data_transforms['train']
    )
    
    print("Loading validation dataset...")
    if args.val_csv_file:
        val_dataset = FairFaceDataset(
            csv_file=args.val_csv_file,
            img_dir=args.img_dir,
            transform=data_transforms['val']
        )
    else:
        # If no separate validation file, split the training data
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    
    # Initialize model
    model = FaceAnalysisModel(
        num_classes_gender=2,
        num_classes_age=9,
        num_classes_race=7
    ).to(device)
    
    # Define loss functions and optimizer
    criterion_gender = nn.CrossEntropyLoss()
    criterion_age = nn.CrossEntropyLoss()
    criterion_race = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    
    # Training loop
    num_epochs = args.epochs
    best_val_loss = float('inf')
    best_model_path = os.path.join(args.output_dir, 'best_model.pth')
    
    # History for plotting
    history = {
        'train_loss': [], 'val_loss': [],
        'train_gender_acc': [], 'val_gender_acc': [],
        'train_age_acc': [], 'val_age_acc': [],
        'train_race_acc': [], 'val_race_acc': []
    }
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_gender_correct = 0
        train_age_correct = 0
        train_race_correct = 0
        train_samples = 0
        
        for images, gender_labels, age_labels, race_labels in tqdm(train_loader, desc="Training"):
            images = images.to(device)
            gender_labels = gender_labels.to(device)
            age_labels = age_labels.to(device)
            race_labels = race_labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            gender_outputs, age_outputs, race_outputs = model(images)
            
            # Calculate loss
            loss_gender = criterion_gender(gender_outputs, gender_labels)
            loss_age = criterion_age(age_outputs, age_labels)
            loss_race = criterion_race(race_outputs, race_labels)
            
            # Combined loss
            loss = loss_gender + loss_age + loss_race
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item() * images.size(0)
            
            _, gender_preds = torch.max(gender_outputs, 1)
            _, age_preds = torch.max(age_outputs, 1)
            _, race_preds = torch.max(race_outputs, 1)
            
            train_gender_correct += torch.sum(gender_preds == gender_labels).item()
            train_age_correct += torch.sum(age_preds == age_labels).item()
            train_race_correct += torch.sum(race_preds == race_labels).item()
            train_samples += images.size(0)
        
        # Calculate average training metrics
        train_loss = train_loss / train_samples
        train_gender_acc = train_gender_correct / train_samples
        train_age_acc = train_age_correct / train_samples
        train_race_acc = train_race_correct / train_samples
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_gender_correct = 0
        val_age_correct = 0
        val_race_correct = 0
        val_samples = 0
        
        with torch.no_grad():
            for images, gender_labels, age_labels, race_labels in tqdm(val_loader, desc="Validation"):
                images = images.to(device)
                gender_labels = gender_labels.to(device)
                age_labels = age_labels.to(device)
                race_labels = race_labels.to(device)
                
                # Forward pass
                gender_outputs, age_outputs, race_outputs = model(images)
                
                # Calculate loss
                loss_gender = criterion_gender(gender_outputs, gender_labels)
                loss_age = criterion_age(age_outputs, age_labels)
                loss_race = criterion_race(race_outputs, race_labels)
                
                # Combined loss
                loss = loss_gender + loss_age + loss_race
                
                # Statistics
                val_loss += loss.item() * images.size(0)
                
                _, gender_preds = torch.max(gender_outputs, 1)
                _, age_preds = torch.max(age_outputs, 1)
                _, race_preds = torch.max(race_outputs, 1)
                
                val_gender_correct += torch.sum(gender_preds == gender_labels).item()
                val_age_correct += torch.sum(age_preds == age_labels).item()
                val_race_correct += torch.sum(race_preds == race_labels).item()
                val_samples += images.size(0)
        
        # Calculate average validation metrics
        val_loss = val_loss / val_samples
        val_gender_acc = val_gender_correct / val_samples
        val_age_acc = val_age_correct / val_samples
        val_race_acc = val_race_correct / val_samples
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Model saved to {best_model_path}")
        
        # Print epoch statistics
        print(f"Train Loss: {train_loss:.4f}, "
              f"Gender Acc: {train_gender_acc:.4f}, "
              f"Age Acc: {train_age_acc:.4f}, "
              f"Race Acc: {train_race_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, "
              f"Gender Acc: {val_gender_acc:.4f}, "
              f"Age Acc: {val_age_acc:.4f}, "
              f"Race Acc: {val_race_acc:.4f}")
        print("-----")
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_gender_acc'].append(train_gender_acc)
        history['val_gender_acc'].append(val_gender_acc)
        history['train_age_acc'].append(train_age_acc)
        history['val_age_acc'].append(val_age_acc)
        history['train_race_acc'].append(train_race_acc)
        history['val_race_acc'].append(val_race_acc)
    
    # Save the final model
    final_model_path = os.path.join(args.output_dir, 'final_model.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    # Plot training history
    plot_training_history(history, args.output_dir)
    
    # Calculate and print overall accuracy
    overall_val_acc = (val_gender_acc + val_age_acc + val_race_acc) / 3
    print(f"Overall validation accuracy: {overall_val_acc:.4f}")
    
    return model, history

def plot_training_history(history, output_dir):
    """
    Plot training and validation metrics.
    """
    plt.figure(figsize=(15, 12))
    
    # Plot loss
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot gender accuracy
    plt.subplot(2, 2, 2)
    plt.plot(history['train_gender_acc'], label='Train Gender Acc')
    plt.plot(history['val_gender_acc'], label='Validation Gender Acc')
    plt.title('Gender Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot age accuracy
    plt.subplot(2, 2, 3)
    plt.plot(history['train_age_acc'], label='Train Age Acc')
    plt.plot(history['val_age_acc'], label='Validation Age Acc')
    plt.title('Age Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot race accuracy
    plt.subplot(2, 2, 4)
    plt.plot(history['train_race_acc'], label='Train Race Acc')
    plt.plot(history['val_race_acc'], label='Validation Race Acc')
    plt.title('Race Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Face Analysis Model')
    parser.add_argument('--csv_file', type=str, required=True, help='Path to the CSV file with training annotations')
    parser.add_argument('--val_csv_file', type=str, default=None, help='Path to the CSV file with validation annotations')
    parser.add_argument('--img_dir', type=str, required=True, help='Directory with the images')
    parser.add_argument('--output_dir', type=str, default='models', help='Directory to save models')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    
    args = parser.parse_args()
    train_model(args)