from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping
from hand_sign_recognize.Scripts.splitData import X_train, X_test, Y_train, Y_test
from hand_sign_recognize.Scripts.checkGPU import check_GPU

# kiểm tra GPU có ss kho6ng

check_GPU()

# Xây dựng mô hình CNN đơn giản
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(36, activation='softmax')
])


def train():
    # Thêm EarlyStopping với patience = 5 (tạm dừng sau 5 epoch nếu độ lỗi không cải thiện)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(
        X_train,
        Y_train,
        epochs=20,
        validation_data=(X_test, Y_test),
        batch_size=32,
        callbacks=[early_stopping]  # Thêm early stopping vào callback
    )

    # Lưu model sau khi huấn luyện xong
    model.save('../Scripts/hand_sign_recognition_cnns.h5')


train()
