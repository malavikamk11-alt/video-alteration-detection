import torch
from torch import nn
from torchvision import models
import os
import ssl  # <--- NEW IMPORT

# --- 1. SSL SECURITY BYPASS (Fixes the Mac Error) ---
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# --- CONFIGURATION ---
MODEL_PATH = "ml_app/deepfake_model_level2.pt" 

# --- MODEL ARCHITECTURE (Level 2: EfficientNet) ---
class Model(nn.Module):
    def __init__(self, num_classes=2, latent_dim=1280, lstm_layers=1, hidden_dim=1280, bidirectional=False):
        super(Model, self).__init__()
        
        # Load EfficientNet-B0
        try:
            model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        except:
            model = models.efficientnet_b0(pretrained=True)
            
        self.model = nn.Sequential(*list(model.children())[:-1])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional=bidirectional)
        self.linear1 = nn.Linear(1280, num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        if x.dim() == 4: x = x.unsqueeze(1)
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        
        fmap = self.model(x)
        x = fmap.view(batch_size, seq_length, 1280)
        x_lstm, _ = self.lstm(x, None)
        return fmap, self.linear1(self.dropout(torch.mean(x_lstm, dim=1)))

if __name__ == "__main__":
    print(f"📂 Loading model from: {MODEL_PATH}")
    
    model = Model(2)
    
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint)
        model.eval()
        print("✅ SUCCESS! The Level 2 Model loaded correctly.")
        print("🚀 You are ready to run the website.")
    except Exception as e:
        print("\n❌ ERROR: Could not load the model.")
        print(f"Details: {e}")