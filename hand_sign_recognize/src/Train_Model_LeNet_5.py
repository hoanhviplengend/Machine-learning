
import os
import sys
import numpy as np
import warnings
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from PIL import Image

warnings.filterwarnings("ignore")
checkpoint = ModelCheckpoint('../Scripts/best_model_lenet.h5', monitor='val_loss', save_best_only=True, mode='min')

# Kiểm tra GPU
if len(tf.config.experimental.list_physical_devices('GPU')) > 0:
    print("Có GPU sẵn sàng.")
else:
    print("Không tìm thấy GPU.")
    sys.exit()


# Hàm để tải dữ liệu từ thư mục
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


# Xây dựng mô hình LeNet-5 đã được điều chỉnh
def train_model_lenet_5(x_train, y_train, x_test, y_test):
    model = Sequential()

    # Lớp tích chập đầu tiên
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3), padding='same'))  # 3 kênh màu
    model.add(BatchNormalization())
    model.add(AveragePooling2D())

    # Lớp tích chập thứ hai
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(AveragePooling2D())

    # Lớp tích chập thứ ba

    # Chuyển đổi đặc trưng thành vector
    model.add(Flatten())

    # model.add(Dense(512, activation='relu'))  # Tăng số lượng nơ-ron
    # model.add(Dropout(0.5))  # Thêm lớp Dropout để giảm overfitting

    # model.add(Dense(256, activation='relu'))  # Tăng số lượng nơ-ron
    # model.add(Dropout(0.5))  # Thêm lớp Dropout để giảm overfitting

    model.add(Dense(128, activation='relu'))  # Tăng số lượng nơ-ron
    model.add(Dropout(0.5))  # Thêm lớp Dropout để giảm overfitting

    # Lớp output với số lớp tương ứng
    model.add(Dense(36, activation='softmax'))  # 36 lớp

    # Biên dịch mô hình
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)
    early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min')

    # Huấn luyện mô hình
    history = model.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test), batch_size=32,
                        callbacks=[checkpoint, reduce_lr, early_stop])

    # Lưu mô hình
    model.save('../Scripts/hand_sign_recognition_lenet5.h5')
    return history


history = train_model_lenet_5(X_train, Y_train, X_test, Y_test)


# Vẽ biểu đồ loss
def draw_chart_loss(history):
    plt.figure(figsize=(12, 4))

    # Loss cho tập huấn luyện
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Loss (Train)')
    plt.plot(history.history['val_loss'], label='Loss (Validation)')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy cho tập huấn luyện
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Accuracy (Train)')
    plt.plot(history.history['val_accuracy'], label='Accuracy (Validation)')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()
draw_chart_loss(history)
