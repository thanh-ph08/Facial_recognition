import torch, os

DB_PATH = "processed_features_cnn.pt"

def remove_person(name_to_remove):
    if not os.path.exists(DB_PATH):
        print(f"File {DB_PATH} không tồn tại!")
        return

    database = torch.load(DB_PATH)
    if name_to_remove in database:
        del database[name_to_remove]
        torch.save(database, DB_PATH)
        print(f"Đã xóa '{name_to_remove}' khỏi database.")
    else:
        print(f"Không tìm thấy '{name_to_remove}' trong database.")

if __name__ == "__main__":
    name = input("Nhập tên người muốn xóa: ").strip()
    remove_person(name)
