import os
import numpy as np
from tensorflow.keras.utils import to_categorical
from PIL import Image


# Hàm để tải dữ liệu từ thư mục và lưu thành file TSV
def create_tsv(data_dir, output_file, num_classes=36):
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
                    images.append(np.array(image).flatten())  # Flatten ảnh
                    labels.append(class_names.index(label))  # Gán nhãn lớp
    # Chuyển nhãn sang định dạng one-hot
    labels_one_hot = to_categorical(labels, num_classes=num_classes)
    # Kết hợp ảnh và nhãn one-hot để lưu
    data = np.hstack((images, labels_one_hot))  # Gộp dữ liệu ảnh và nhãn vào cùng một mảng
    data = np.array(data, dtype=np.float32)
    # Tạo tiêu đề
    header = "\t".join(
        [f"Pixel_{i}" for i in range(images[0].size)] +
        [f"Label_{i}" for i in range(num_classes)]
    )
    # Lưu thành file TSV
    np.savetxt(output_file, data, delimiter="\t", fmt="%.0f", header=header, comments="")
    print(f"Dữ liệu đã được lưu vào {output_file}")


# Hàm để tải dữ liệu từ file TSV
def load_from_tsv(tsv_file, img_shape=(32, 32, 3), num_classes=36):
    data = np.loadtxt(tsv_file, delimiter="\t", skiprows=1)  # Bỏ qua dòng tiêu đề
    images = data[:, :-num_classes].reshape(-1, *img_shape)  # Lấy các giá trị pixel, reshape thành ảnh
    labels = data[:, -num_classes:]  # Lấy nhãn ở định dạng one-hot
    return images, labels


# Đường dẫn đến thư mục dữ liệu
train_dir = r'..\Data\Gesture Image Pre-Processed Data'
test_dir = r'..\Data\Gesture Image Pre-Processed Data - Test'
# Tạo file TSV cho tập train và test
train_tsv = r"..\Data\train_data.tsv"
test_tsv = r"..\Data\test_data.tsv"
create_tsv(train_dir, train_tsv)
create_tsv(test_dir, test_tsv)

# Hàm để tải dữ liệu từ thư mục
# def load_data(data_dir):
#     images = []
#     labels = []
#     class_names = os.listdir(data_dir)  # Lấy tên các lớp từ thư mục
#
#     for label in class_names:
#         class_dir = os.path.join(data_dir, label)
#         if os.path.isdir(class_dir):  # Kiểm tra xem đây có phải là thư mục không
#             for filename in os.listdir(class_dir):
#                 if filename.endswith('.png') or filename.endswith('.jpg'):  # Kiểm tra định dạng tệp
#                     img_path = os.path.join(class_dir, filename)
#                     image = Image.open(img_path).convert('RGB')  # Chuyển đổi thành ảnh RGB
#                     image = image.resize((32, 32))  # Thay đổi kích thước ảnh
#                     images.append(np.array(image))
#                     labels.append(class_names.index(label))  # Gán nhãn lớp
#
#     return np.array(images), to_categorical(np.array(labels), num_classes=36)  # Trả về mảng NumPy của ảnh và nhãn
