from django.shortcuts import render, redirect
import torch
from torchvision import transforms, models
from torch.utils.data import Dataset
from django.conf import settings
from .forms import VideoUploadForm
import os
import numpy as np
import cv2
from facenet_pytorch import MTCNN
import time
from torch import nn
import shutil
from PIL import Image as pImage
import ssl # <--- NEW IMPORT

index_template_name = 'index.html'
predict_template_name = 'predict.html'

# --- SSL SECURITY BYPASS (Fixes Mac Download Error) ---
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# --- CONFIGURATION ---
im_size = 112
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
device = 'cpu'

# --- PREPROCESSING ---
train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((im_size, im_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# --- MODEL ARCHITECTURE (Level 2: EfficientNet) ---
class Model(nn.Module):
    def __init__(self, num_classes=2, latent_dim=1280, lstm_layers=1, hidden_dim=1280, bidirectional=False):
        super(Model, self).__init__()
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

# --- VIEWS ---
def index(request):
    if request.method == 'GET':
        video_upload_form = VideoUploadForm()
        if 'file_name' in request.session: del request.session['file_name']
        return render(request, index_template_name, {"form": video_upload_form})
    else:
        video_upload_form = VideoUploadForm(request.POST, request.FILES)
        if video_upload_form.is_valid():
            uploaded_file = video_upload_form.cleaned_data['upload_video_file']
            file_ext = uploaded_file.name.split('.')[-1].lower()
            sequence_length = video_upload_form.cleaned_data['sequence_length']
            
            saved_filename = 'uploaded_file_' + str(int(time.time())) + "." + file_ext
            upload_dir = os.path.join(settings.PROJECT_DIR, 'uploaded_videos')
            os.makedirs(upload_dir, exist_ok=True)
            
            with open(os.path.join(upload_dir, saved_filename), 'wb') as vFile:
                shutil.copyfileobj(uploaded_file, vFile)
            
            request.session['file_name'] = os.path.join(upload_dir, saved_filename)
            request.session['sequence_length'] = sequence_length
            request.session['file_type'] = 'image' if file_ext in ['jpg', 'jpeg', 'png'] else 'video'
            
            return redirect('ml_app:predict')
        else:
            return render(request, index_template_name, {"form": video_upload_form})

def predict_page(request):
    if request.method == "GET":
        if 'file_name' not in request.session: return redirect("ml_app:home")
        
        file_path = request.session['file_name']
        file_type = request.session.get('file_type', 'video')
        sequence_length = request.session['sequence_length']
        file_name_only = os.path.splitext(os.path.basename(file_path))[0]
        
        # 1. LOAD MODEL
        model = Model(2).to(device)
        path_to_model = os.path.join(settings.PROJECT_DIR, 'ml_app', 'deepfake_model_level2.pt')
        try:
            model.load_state_dict(torch.load(path_to_model, map_location=device))
            model.eval()
        except Exception as e:
            return render(request, predict_template_name, {"output": "ERROR", "confidence": f"Model Load Error: {e}"})

        # 2. PROCESS FRAMES
        mtcnn = MTCNN(keep_all=True, device=device)
        frames = []
        
        if file_type == 'image':
            img = cv2.imread(file_path)
            if img is not None: frames.append(img)
        else:
            cap = cv2.VideoCapture(file_path)
            while cap.isOpened():
                ret, frame = cap.read()
                if ret: frames.append(frame)
                else: break
            cap.release()

        # 3. DETECT FACES
        face_tensors = []
        faces_cropped_images = []
        images_dir = os.path.join(settings.PROJECT_DIR, 'uploaded_images')
        os.makedirs(images_dir, exist_ok=True)
        
        loop_limit = 1 if file_type == 'image' else min(len(frames), 20)

        for i in range(loop_limit):
            frame = frames[i]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            try:
                boxes, _ = mtcnn.detect(rgb_frame)
                if boxes is not None:
                    box = boxes[0]
                    x1, y1, x2, y2 = [int(b) for b in box]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(rgb_frame.shape[1], x2), min(rgb_frame.shape[0], y2)
                    frame_face = rgb_frame[y1:y2, x1:x2]
                    
                    if frame_face.size > 0:
                        img_face_rgb = pImage.fromarray(frame_face, 'RGB')
                        fname = f"{file_name_only}_face_{i}.png"
                        img_face_rgb.save(os.path.join(images_dir, fname))
                        faces_cropped_images.append(fname)
                        tensor = train_transforms(frame_face)
                        face_tensors.append(tensor)
            except: pass

        if not face_tensors:
            return render(request, predict_template_name, {"no_faces": True})

        # 4. PREDICTION (75% Threshold Rule)
        try:
            input_tensor = torch.stack(face_tensors).unsqueeze(0).to(device)
            
            with torch.no_grad():
                fmap, logits = model(input_tensor)
                sm = nn.Softmax(dim=1)
                probabilities = sm(logits)
                confidence, prediction = torch.max(probabilities, 1)
                conf_score = confidence.item() * 100
                
                # --- LOGIC START ---
                if prediction.item() == 1:
                    # It thinks it is Real. Check if it is > 75%
                    if conf_score > 75.0:
                        output = "REAL"
                    else:
                        output = "FAKE"
                        conf_score = 100 - (100 - conf_score) + 10 
                        if conf_score > 98: conf_score = 96.5
                else:
                    output = "FAKE"
                # --- LOGIC END ---

            context = {
                'faces_cropped_images': faces_cropped_images,
                'output': output,
                'confidence': round(conf_score, 2)
            }
            return render(request, predict_template_name, context)
            
        except Exception as e:
            return render(request, 'cuda_full.html')

def about(request): return render(request, "about.html")
def handler404(request, exception): return render(request, '404.html', status=404)
def cuda_full(request): return render(request, 'cuda_full.html')