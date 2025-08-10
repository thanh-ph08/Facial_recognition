import cv2
import torch
import numpy as np
import random
from PIL import Image
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def get_embedding(model, frame, device):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    img_tensor = transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(img_tensor)
    return embedding.cpu().numpy().flatten()


def recognize_face(model, database, frame, num_samples=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedding = get_embedding(model, frame, device)
    
    best_match = None
    min_dist = float('inf')

    for person_id, features in database.items():
        features = np.array(features)  # Đảm bảo là numpy array

        # Random chọn tối đa `num_samples` embedding
        if len(features) > num_samples:
            selected = features[random.sample(range(len(features)), num_samples)]
        else:
            selected = features  # Nếu ít hơn thì lấy hết

        dists = np.linalg.norm(selected - embedding, axis=1)
        dist = np.mean(dists)

        if dist < min_dist:
            min_dist = dist
            best_match = person_id

    # Ngưỡng nhận diện (tùy chỉnh)
    if min_dist < 0.19:
        return best_match
    else:
        return "Unknown"
