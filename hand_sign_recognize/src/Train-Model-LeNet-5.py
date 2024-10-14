import sys
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

checkpoint = ModelCheckpoint('..\Scripts\best_model_lenet.h5', monitor='val_loss', save_best_only=True, mode='min')

# Kiểm tra GPU
if len(tf.config.experimental.list_physical_devices('GPU')) > 0:
    print("Có GPU sẵn sàng.")
else:
    print("Không tìm thấy GPU.")
    sys.exit()

# Đường dẫn đến thư mục dữ liệu
train_dir = r'..\Data\Train'
test_dir = r'..\Data\Test'

# Thiết lập các tham số
image_size = (28, 28)  # Kích thước ảnh phù hợp với LeNet-5 (28x28)
batch_size = 64  # Kích thước batch
num_classes = 24  # Số lượng lớp (24 ký tự từ A đến Z, bỏ qua 'J' và 'Z')

# Tạo bộ sinh dữ liệu cho tập huấn luyện và tập kiểm tra
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255, rotation_range=20, width_shift_range=0.2,
    height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,
    horizontal_flip=True, fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Xây dựng mô hình LeNet-5 đã được điều chỉnh
model = Sequential()

# Lớp tích chập đầu tiên
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 3), padding='same'))
model.add(BatchNormalization())
model.add(AveragePooling2D())

# Lớp tích chập thứ hai
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(AveragePooling2D())

# Lớp tích chập thứ ba (thêm lớp tích chập thứ ba)
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(AveragePooling2D())

# Chuyển đổi đặc trưng thành vector
model.add(Flatten())

# Lớp fully connected
model.add(Dense(256, activation='relu'))  # Tăng số lượng nơ-ron
model.add(Dropout(0.5))  # Thêm lớp Dropout để giảm overfitting
model.add(Dense(128, activation='relu'))  # Tăng số lượng nơ-ron
model.add(Dropout(0.5))  # Thêm lớp Dropout để giảm overfitting

# Lớp output với số lớp tương ứng
model.add(Dense(num_classes, activation='softmax'))

# Biên dịch mô hình
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)
early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min')

# Huấn luyện mô hình
history = model.fit(train_generator, epochs=50, validation_data=test_generator, callbacks=[checkpoint, reduce_lr, early_stop])

# Lưu mô hình
model.save('..\Scripts\hand_sign_recognition_lenet5.h5')

# Vẽ biểu đồ loss
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
