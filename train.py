import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import cv2
import os
import glob
import sys
import time
import random

# --- CONFIGURATION ---
# Path to the CSV file (Ensure this file exists in this path!)
CSV_PATH = os.path.join("Model Creation", "labels", "Gobal_metadata.csv")

# Path to the videos you just processed
DATA_DIR = "Processed_Data"

# Where to save the trained model
CHECKPOINT_PATH = "deepfake_model_m2.pt"

# Training Settings
EPOCHS = 20
BATCH_SIZE = 8     # Reduced for M2 Air memory
IM_SIZE = 112
SEQUENCE_LENGTH = 20 # Frames to analyze per video (reduced from 100 to save memory/time)

# --- DEVICE SETUP (M2 Optimization) ---
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Training on device: {device}")

# --- DATASET CLASS ---
class VideoDataset(Dataset):
    def __init__(self, video_files, labels_df, sequence_length=20, transform=None):
        self.video_files = video_files
        self.labels = labels_df
        self.sequence_length = sequence_length
        self.transform = transform

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        video_name = os.path.basename(video_path)
        
        # Get Label from CSV
        try:
            # Find the row where 'file' matches the video name
            row = self.labels.loc[self.labels["file"] == video_name]
            if row.empty:
                # If not found in CSV, assume fake if in synthesis folder, else real
                # This is a fallback
                label_str = 'FAKE' if 'synthesis' in video_path else 'REAL'
            else:
                label_str = row.iloc[0]["label"]
        except Exception as e:
             label_str = 'FAKE' # Default fallback

        label = 0 if label_str == 'FAKE' else 1

        # Extract Frames
        frames = self.frame_extract(video_path)
        
        # Pad or cut frames to match sequence_length
        frames = frames[:self.sequence_length]
        
        # Transform frames
        processed_frames = []
        for frame in frames:
            if self.transform:
                frame = self.transform(frame)
            processed_frames.append(frame)
        
        # If video was too short, pad with zeros
        while len(processed_frames) < self.sequence_length:
            processed_frames.append(torch.zeros(3, IM_SIZE, IM_SIZE))

        # Stack into a tensor (Sequence_Length, Channels, Height, Width)
        data = torch.stack(processed_frames)
        
        return data, label

    def frame_extract(self, path):
        vidObj = cv2.VideoCapture(path)
        success = 1
        frames = []
        while success:
            success, image = vidObj.read()
            if success:
                # Convert BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                frames.append(image)
        return frames

# --- MODEL ARCHITECTURE (ResNeXt + LSTM) ---
class ResNextLSTM(nn.Module):
    def __init__(self, num_classes=2, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(ResNextLSTM, self).__init__()
        
        # Load Pretrained ResNeXt
        model = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1)
        
        # Remove the last two layers (Pooling and FC) to get features
        self.model = nn.Sequential(*list(model.children())[:-2])
        
        # LSTM Layer
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional=bidirectional)
        
        # Classification Layers
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        
        # Pass through CNN (ResNeXt)
        # Merge batch and sequence dimensions to process all frames at once
        x = x.view(batch_size * seq_length, c, h, w)
        
        fmap = self.model(x)
        x = self.avgpool(fmap)
        
        # Reshape back to (Batch, Sequence, Features)
        x = x.view(batch_size, seq_length, 2048)
        
        # Pass through LSTM
        x_lstm, _ = self.lstm(x, None)
        
        # Mean pooling across the sequence dimension (temporal averaging)
        return self.dp(self.linear1(torch.mean(x_lstm, dim=1)))

# --- HELPER CLASS ---
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0; self.avg = 0; self.sum = 0; self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def calculate_accuracy(outputs, targets):
    batch_size = targets.size(0)
    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))
    n_correct_elems = correct.float().sum().item()
    return 100 * n_correct_elems / batch_size

# --- MAIN TRAINING FUNCTION ---
def train_epoch(epoch, num_epochs, data_loader, model, criterion, optimizer):
    model.train()
    losses = AverageMeter()
    accuracies = AverageMeter()
    
    for i, (inputs, targets) in enumerate(data_loader):
        # Move data to M2 GPU
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # Forward pass
        outputs = model(inputs)
        
        # Calculate loss
        loss = criterion(outputs, targets)
        acc = calculate_accuracy(outputs, targets)
        
        losses.update(loss.item(), inputs.size(0))
        accuracies.update(acc, inputs.size(0))
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [Loss: %f, Acc: %.2f%%]"
            % (epoch, num_epochs, i, len(data_loader), losses.avg, accuracies.avg)
        )
    return losses.avg, accuracies.avg

# --- SCRIPT EXECUTION ---
if __name__ == "__main__":
    # 1. Load Metadata
    print("Loading Labels...")
    try:
        labels_df = pd.read_csv(CSV_PATH)
        print(f"Labels loaded. Found {len(labels_df)} entries.")
    except FileNotFoundError:
        print(f"ERROR: Could not find CSV at {CSV_PATH}")
        print("Please check the folder structure.")
        sys.exit()

    # 2. Find Video Files
    print("Looking for video files...")
    # Look recursively for all mp4 files in Processed_Data
    video_files = glob.glob(os.path.join(DATA_DIR, "*", "*.mp4"))
    
    if not video_files:
        print(f"ERROR: No videos found in {DATA_DIR}")
        print("Make sure you ran the preprocessing script first!")
        sys.exit()
        
    random.shuffle(video_files)
    print(f"Found {len(video_files)} processed videos.")

    # 3. Split Data (80% Train, 20% Validation)
    split_ratio = 0.8
    split_idx = int(len(video_files) * split_ratio)
    train_videos = video_files[:split_idx]
    valid_videos = video_files[split_idx:]
    
    print(f"Training on {len(train_videos)} videos, Validating on {len(valid_videos)} videos.")

    # 4. Define Transforms
    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IM_SIZE, IM_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 5. Create DataLoaders
    train_data = VideoDataset(train_videos, labels_df, sequence_length=SEQUENCE_LENGTH, transform=train_transforms)
    val_data = VideoDataset(valid_videos, labels_df, sequence_length=SEQUENCE_LENGTH, transform=train_transforms)
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    valid_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 6. Initialize Model
    print("Initializing Model...")
    model = ResNextLSTM(num_classes=2).to(device)
    
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)

    # 7. Start Training Loop
    print("Starting Training...")
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(epoch, EPOCHS, train_loader, model, criterion, optimizer)
        print(f"\nEpoch {epoch} Completed. Avg Loss: {train_loss:.4f}, Avg Acc: {train_acc:.2f}%")
        
        # Save checkpoint every epoch
        torch.save(model.state_dict(), CHECKPOINT_PATH)
        print(f"Model saved to {CHECKPOINT_PATH}")

    print("Training Complete!")