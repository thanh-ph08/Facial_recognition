import os
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.datasets import fetch_lfw_people
from PIL import Image

# ------------ Config ------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ------------ Load LFW Dataset ------------
lfw = fetch_lfw_people(min_faces_per_person=5, resize=0.5, color=True)
images = lfw.images
labels = lfw.target
num_classes = len(lfw.target_names)
print(f"LFW: {len(images)} images, {num_classes} classes")

# ------------ Dataset ------------
class LFWDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            img = self.transform(Image.fromarray((img * 255).astype('uint8')))
        return img, label

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

dataset = LFWDataset(images, labels, transform=transform)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# ------------ CNN Params (khởi tạo thủ công) ------------
conv_w = torch.randn(32, 3, 3, 3, device=DEVICE) * 0.01  # out=32 channels, in=3, 3x3 kernel
conv_b = torch.zeros(32, device=DEVICE)

fc_w = torch.randn(32 * 64 * 64, 128, device=DEVICE) * 0.01  # Flatten đầu vào
fc_b = torch.zeros(128, device=DEVICE)

for param in [conv_w, conv_b, fc_w, fc_b]:
    param.requires_grad_()

# ------------ Forward CNN ------------
def forward_cnn(x, conv_w, conv_b, fc_w, fc_b):
    x = F.conv2d(x, conv_w, conv_b, stride=1, padding=1)  # giữ nguyên size
    x = F.relu(x)
    x = F.max_pool2d(x, kernel_size=2)  # giảm nửa size: (128 → 64)
    x = x.view(x.size(0), -1)  # flatten
    x = torch.matmul(x, fc_w) + fc_b
    x = F.normalize(x, p=2, dim=1)
    return x

# ------------ Batch Hard Triplet Loss ------------
def batch_hard_triplet_loss(embs, labels, margin=1.0):
    dist = torch.cdist(embs, embs)
    mask_pos = labels.unsqueeze(1) == labels.unsqueeze(0)
    mask_neg = ~mask_pos
    hardest_pos = torch.max(dist * mask_pos.float(), dim=1)[0]
    max_neg = torch.max(dist, dim=1, keepdim=True)[0]
    dist_neg = dist + max_neg * (~mask_neg).float()
    hardest_neg = torch.min(dist_neg, dim=1)[0]
    loss = torch.clamp(hardest_pos - hardest_neg + margin, min=0.0)
    return loss.mean()

# ------------ Optimizer ------------
optimizer = optim.Adam([conv_w, conv_b, fc_w, fc_b], lr=1e-3)

# ------------ Training Loop ------------
for epoch in range(20):
    total_loss = 0
    start = time.time()
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
        embs = forward_cnn(imgs, conv_w, conv_b, fc_w, fc_b)
        loss = batch_hard_triplet_loss(embs, lbls)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}: Loss = {total_loss/len(loader):.4f}, Time = {time.time()-start:.1f}s")

# ------------ Save weights ------------
torch.save({
    "conv_w": conv_w.cpu(),
    "conv_b": conv_b.cpu(),
    "fc_w": fc_w.cpu(),
    "fc_b": fc_b.cpu()
}, "cnn_functional_model.pt")
print("✅ Saved model to cnn_functional_model.pt")
