import os
import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# ------------------ Cấu hình ------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Đang sử dụng thiết bị: {DEVICE}")

MODEL_PATH = "cnn_functional_model.pt"
DB_PATH = "face_database.pt"
THRESHOLD = 0.9  # Ngưỡng nhận diện, có thể điều chỉnh
MAX_IMAGES_PER_PERSON = 20  # Giới hạn số ảnh cho mỗi người

# ------------------ Tải trọng số ------------------
state = torch.load(MODEL_PATH, map_location=DEVICE)
conv_w = state["conv_w"].to(DEVICE)
conv_b = state["conv_b"].to(DEVICE)
fc_w = state["fc_w"].to(DEVICE)
fc_b = state["fc_b"].to(DEVICE)

# ------------------ Mạng CNN thuần hàm ------------------
def forward_cnn(x, conv_w, conv_b, fc_w, fc_b):
    x = F.conv2d(x, conv_w, conv_b, stride=1, padding=1)
    x = F.relu(x)
    x = F.max_pool2d(x, kernel_size=2)
    x = x.view(x.size(0), -1)
    x = torch.matmul(x, fc_w) + fc_b
    x = F.normalize(x, p=2, dim=1)
    return x

# ------------------ Load cơ sở dữ liệu ------------------
if os.path.exists(DB_PATH):
    database = torch.load(DB_PATH)
    print(f"📁 Đã tải database với {len(database)} người.")
else:
    database = {}
    print("📁 Không tìm thấy database, tạo mới.")

# ------------------ Tiền xử lý ảnh ------------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ------------------ Phát hiện khuôn mặt ------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ------------------ Hàm trích xuất embedding ------------------
def extract_embedding(face_img_bgr):
    img_rgb = Image.fromarray(cv2.cvtColor(face_img_bgr, cv2.COLOR_BGR2RGB))
    tensor = transform(img_rgb).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        emb = forward_cnn(tensor, conv_w, conv_b, fc_w, fc_b).squeeze(0)
    return emb

# ------------------ Hàm nhận diện ------------------
def recognize(query_emb):
    best_name, best_dist = "Unknown", float('inf')
    for name, tensors in database.items():
        tensors = tensors.to(DEVICE)
        with torch.no_grad():
            embs = forward_cnn(tensors, conv_w, conv_b, fc_w, fc_b)
        dists = torch.norm(embs - query_emb.unsqueeze(0), dim=1)
        min_dist = torch.min(dists).item()
        if min_dist < best_dist:
            best_name, best_dist = name, min_dist
    print(f"🔍 Nhận diện: {best_name} (khoảng cách {best_dist:.4f})")
    return (best_name, best_dist) if best_dist < THRESHOLD else ("Unknown", best_dist)

def get_all_names():
    return list(database.keys())

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    print("📷 Đang mở camera. Nhấn 'a' để thêm người, ESC để thoát.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Không thể đọc từ camera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            emb = extract_embedding(face_img)
            name, dist = recognize(emb)
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            label = f"{name} ({dist:.2f})" if name != "Unknown" else "Unknown"
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

        elif key == ord('a'):  # Thêm người mới
            new_name = input("✏️ Nhập tên người mới: ").strip()
            if not new_name:
                print("❌ Tên không hợp lệ.")
                continue

            if len(faces) == 0:
                print("❌ Không phát hiện khuôn mặt để thêm.")
                continue

            added_count = 0
            for (x, y, w, h) in faces:
                face_img = frame[y:y+h, x:x+w]
                img_rgb = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
                tensor = transform(img_rgb).unsqueeze(0).cpu()

                if new_name in database:
                    if database[new_name].shape[0] >= MAX_IMAGES_PER_PERSON:
                        print(f"⚠️ {new_name} đã đủ {MAX_IMAGES_PER_PERSON} ảnh.")
                        break
                    database[new_name] = torch.cat([database[new_name], tensor], dim=0)
                else:
                    database[new_name] = tensor
                added_count += 1

            if added_count > 0:
                torch.save(database, DB_PATH)
                print(f"✅ Đã thêm {added_count} ảnh cho {new_name}.")
            else:
                print("⚠️ Không thêm được ảnh nào.")

        cv2.imshow("Real-time Face Recognition", frame)

    cap.release()
    cv2.destroyAllWindows()

