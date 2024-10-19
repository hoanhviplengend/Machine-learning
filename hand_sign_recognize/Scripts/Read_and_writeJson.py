import json
import os

def read_json(file_path):
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:  # Kiểm tra nếu file tồn tại và không trống
        with open(file_path, 'r') as json_file:
            try:
                return json.load(json_file)
            except json.JSONDecodeError:
                return []
    return []

# Hàm ghi dữ liệu vào file JSON
def write_json(file_path, data):
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)
