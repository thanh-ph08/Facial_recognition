import os
import time
import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN
from model import load_cnn, load_mlp, load_embeddings
from recognize import get_embedding
from PIL import Image
from torchvision import transforms
import pickle
import random

# =====================
# Cấu hình & khởi tạo
# =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CNN_WEIGHTS = "cnn_weights.pth"
MLP_WEIGHTS = "mlp_weights.pkl"
DB_PATH = "processed_features_cnn.pt"

cnn_model = load_cnn(CNN_WEIGHTS)
W1, b1, W2, b2 = load_mlp(MLP_WEIGHTS)
database = load_embeddings(DB_PATH) if os.path.exists(DB_PATH) else {}

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

mtcnn = MTCNN(image_size=128, margin=20, keep_all=True, device=DEVICE)


# =====================
# Hàm chuyển embedding CNN qua MLP lấy vector 128 chiều
# =====================
def get_mlp_embedding(embedding_input):
    if embedding_input.ndim > 1:
        embedding_input = embedding_input.flatten()

    h1 = np.dot(embedding_input, W1) + b1
    h1 = np.maximum(h1, 0)  # ReLU
    out = np.dot(h1, W2) + b2  # Output embedding 128-dim
    return out


# =====================
# Hàm nhận diện dùng khoảng cách Euclidean với mẫu trong DB (Siamese)
# Mỗi người trong DB giữ list embedding (vector 128-dim)
# Mỗi lần nhận diện random lấy 3 ảnh đại diện của mỗi người để tính khoảng cách trung bình
# =====================
def recognize_face_mlp(embedding):
    query_emb = get_mlp_embedding(embedding)
    min_dist = float('inf')
    best_label = "Unknown"
    threshold = 1.5  # Ngưỡng khoảng cách cho nhận diện (có thể điều chỉnh)

    for person, vec_list in database.items():
        if len(vec_list) == 0:
            continue
        # Random 3 embedding đại diện hoặc ít hơn nếu không đủ
        sampled_vecs = random.sample(list(vec_list), min(len(vec_list), 3))
        # Tính khoảng cách Euclidean trung bình
        dists = [np.linalg.norm(query_emb - get_mlp_embedding(v)) for v in sampled_vecs]
        avg_dist = np.mean(dists)

        if avg_dist < min_dist:
            min_dist = avg_dist
            best_label = person

    if min_dist > threshold:
        return "Unknown", min_dist
    else:
        return best_label, min_dist


# =====================
# Thêm người mới
# =====================
def add_new_person(name, face_img):
    embedding = get_embedding(cnn_model, face_img, DEVICE)
    if name in database:
        database[name].append(embedding)
    else:
        database[name] = [embedding]
    torch.save(database, DB_PATH)
    print(f"[INFO] Đã thêm {name} vào database.")


# =====================
# Chạy demo webcam
# =====================
def run():
    cap = cv2.VideoCapture(0)
    print("[INFO] Nhấn SPACE để chụp & nhận diện | 'c' để thêm người mới | 'q' để thoát")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Hiển thị camera nhưng không xử lý cho đến khi bấm SPACE
        cv2.imshow("Face Recognition", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord(" "):  # Nhấn SPACE để nhận diện khuôn mặt
            boxes, _ = mtcnn.detect(frame)

            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

                    if x2 <= x1 or y2 <= y1:
                        continue

                    face_img = frame[y1:y2, x1:x2]
                    try:
                        face_resized = cv2.resize(face_img, (128, 128))
                    except:
                        continue

                    embedding = get_embedding(cnn_model, face_resized, DEVICE)
                    name, dist = recognize_face_mlp(embedding)
                    display_name = f"{name} ({dist:.2f})"

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, display_name, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                    cv2.imshow("Face Recognition", frame)
                    cv2.waitKey(0)  # Đợi phím bất kỳ để tiếp tục

            else:
                print("[INFO] Không phát hiện khuôn mặt.")

        elif key == ord("c"):
            boxes, _ = mtcnn.detect(frame)
            if boxes is not None and len(boxes) > 0:
                x1, y1, x2, y2 = map(int, boxes[0])
                face_img = frame[y1:y2, x1:x2]
                face_img = cv2.resize(face_img, (128, 128))
                new_name = input("Tên người mới: ").strip()
                if new_name:
                    add_new_person(new_name, face_img)

    cap.release()
    cv2.destroyAllWindows()


# =====================
# Main
# =====================
if __name__ == "__main__":
    run()
