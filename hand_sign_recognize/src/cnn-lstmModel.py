import numpy as np
from tensorflow.keras import layers, models
from hand_sign_recognize.Scripts.splitData import X_train, X_test, Y_train, Y_test
from hand_sign_recognize.Scripts.checkGPU import check_GPU

check_GPU()

# Giả sử bạn đã có dữ liệu được tải vào X_train, Y_train, X_test, Y_test
# X_train: (num_samples, 30, 32, 32, 3)
# Y_train: (num_samples, 36) -> số lớp là 36
# Y_test: (num_samples, 36)

# Chuyển đổi nhãn sang dạng one-hot encoding
num_samples = X_train.shape[0]
X_train = np.repeat(X_train[np.newaxis, :], 30, axis=0)  # Thêm 30 khung hình cho mỗi mẫu
X_train = X_train.reshape((num_samples, 30, 32, 32, 3))  # Đảm bảo đúng hình dạng

# Xây dựng mô hình CNN-LSTM
model = models.Sequential([
    layers.TimeDistributed(layers.Conv2D(32, (3, 3), activation='relu'), input_shape=(30, 32, 32, 3)),
    layers.TimeDistributed(layers.MaxPooling2D((2, 2))),
    layers.TimeDistributed(layers.Conv2D(64, (3, 3), activation='relu')),
    layers.TimeDistributed(layers.MaxPooling2D((2, 2))),
    layers.TimeDistributed(layers.Conv2D(128, (3, 3), activation='relu')),
    layers.TimeDistributed(layers.MaxPooling2D((2, 2))),
    layers.TimeDistributed(layers.Flatten()),
    layers.LSTM(128, return_sequences=True),
    layers.LSTM(64),
    layers.Dense(36, activation='softmax')  # Số lớp đầu ra
])

# Biên dịch mô hình
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Huấn luyện mô hình
history = model.fit(X_train, Y_train, epochs=20, batch_size=8, validation_data=(X_test, Y_test))

# Lưu mô hình
model.save('hand_sign_recognition_model.h5')
