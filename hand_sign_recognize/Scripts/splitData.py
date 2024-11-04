import os
import numpy as np
from tensorflow.keras.utils import to_categorical
from PIL import Image


def load_data(data_dir):
    images = []
    labels = []
    class_names = os.listdir(data_dir)  # Lấy tên các lớp từ thư mục

    for label in class_names:
        class_dir = os.path.join(data_dir, label)
        if os.path.isdir(class_dir):  # Kiểm tra xem đây có phải là thư mục không
            for filename in os.listdir(class_dir):
                if filename.endswith('.png') or filename.endswith('.jpg'):  # Kiểm tra định dạng tệp
                    img_path = os.path.join(class_dir, filename)
                    image = Image.open(img_path).convert('RGB')  # Chuyển đổi thành ảnh RGB
                    image = image.resize((32, 32))  # Thay đổi kích thước ảnh
                    images.append(np.array(image))
                    labels.append(class_names.index(label))  # Gán nhãn lớp

    return np.array(images), to_categorical(np.array(labels), num_classes=36)  # Trả về mảng NumPy của ảnh và nhãn


# Đường dẫn đến thư mục dữ liệu
train_dir = r'..\Data\Gesture Image Pre-Processed Data'
test_dir = r'..\Data\Gesture Image Pre-Processed Data - Test'


# Tải dữ liệu
X_train, Y_train = load_data(train_dir)
X_test, Y_test = load_data(test_dir)

# Đảm bảo dữ liệu có định dạng đúng
X_train = X_train.astype('float32') / 255  # Chuẩn hóa dữ liệu
X_test = X_test.astype('float32') / 255



