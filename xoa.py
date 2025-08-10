import torch

db_path = "processed_features_cnn.pt"

# Load database (weights_only=False để tránh lỗi PyTorch 2.6+)
database = torch.load(db_path, weights_only=False)

# Danh sách người cần xóa
del database["Vhuy"]

# Lưu lại database sau khi xóa
torch.save(database, db_path)
print("✅ Đã lưu lại database sau khi xóa.")
