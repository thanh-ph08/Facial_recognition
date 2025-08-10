import torch
import torch.nn as nn
import numpy as np
import pickle

# ======================
# CNN Feature Extractor
# ======================
class FeatureExtractorCNN(nn.Module):
    def __init__(self):
        super(FeatureExtractorCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        return x

# ======================
# Load CNN weights
# ======================
def load_cnn(weights_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FeatureExtractorCNN().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model

# ======================
# Load MLP weights
# ======================
def load_mlp(weights_path):
    with open(weights_path, "rb") as f:
        weights = pickle.load(f)
    return weights["W1"], weights["b1"], weights["W2"], weights["b2"]

# ======================
# Load embedding database
# ======================
def load_embeddings(db_path):
    return torch.load(db_path, weights_only=False)
