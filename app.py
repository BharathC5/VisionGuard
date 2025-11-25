from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
from PIL import Image
import io
import base64
import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models

# ---------------- FLASK SETUP ----------------
app = Flask(__name__, static_url_path='/static')
CORS(app)

# ---------------- MODEL DEFINITION (matches neural_ga_fuzzy.py) ----------------
class NeuroBackbone(nn.Module):
    def __init__(self, num_classes: int = 5, feature_dim: int = 1024,
                 dropout: float = 0.3, pretrained: bool = False):
        super().__init__()
        # same as in neural_ga_fuzzy.py
        try:
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            base = models.efficientnet_b0(weights=weights)
        except Exception:
            base = models.efficientnet_b0(weights=None)

        in_features = base.classifier[1].in_features
        base.classifier = nn.Identity()
        self.backbone = base

        self.feature_proj = nn.Sequential(
            nn.Linear(in_features, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        feats = self.feature_proj(x)
        logits = self.classifier(feats)
        return logits, feats

# ---------------- LOAD .pt MODEL ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = None
try:
    model = NeuroBackbone(num_classes=5, feature_dim=1024, dropout=0.3, pretrained=False)
    # adjust path if your file is elsewhere
    state_dict = torch.load("model/best_model.pt", map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("PyTorch model loaded successfully!")
except Exception as e:
    print("Error loading model:")
    print(e)
    print(traceback.format_exc())
    model = None  # so route can detect failure

# ---------------- PREPROCESS (match training: img_size=300) ----------------
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ---------------- ROUTES ----------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/home', methods=['POST'])
def predict_retinopathy():
    try:
        if model is None:
            # if this happens, check your terminal for the real load error
            return jsonify({"error": "Model not loaded on server"}), 500

        print("Received prediction request")

        image_base64 = request.json['image']
        print("Image base64 received")

        # strip data URL prefix if present
        if "base64," in image_base64:
            image_base64 = image_base64.split("base64,")[1]

        # decode image
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        print("Image processed successfully")

        # preprocess
        tensor = transform(image).unsqueeze(0).to(device)

        # predict
        with torch.no_grad():
            logits, feats = model(tensor)
            probs = F.softmax(logits, dim=1)
            stage = int(torch.argmax(probs, dim=1).item())
            confidence = float(probs[0, stage].item())

        print(f"Prediction result: stage={stage}, confidence={confidence:.4f}")

        return jsonify({
            "stage": stage,
            "confidence": confidence
        })

    except Exception as e:
        print("Prediction Error:")
        print(traceback.format_exc())
        return jsonify({
            "error": str(e),
            "details": traceback.format_exc()
        }), 400


if __name__ == '__main__':
    app.run(debug=True, port=5000)
