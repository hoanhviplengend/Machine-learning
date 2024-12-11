from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping
from hand_sign_recognize.Scripts.splitData import X_train, X_test, Y_train, Y_test
from hand_sign_recognize.Scripts.checkGPU import check_GPU

check_GPU()

# Tải ResNet50 pre-trained với ImageNet
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# Thêm các lớp để điều chỉnh cho bài toán của bạn
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Tạo đầu ra 1 chiều từ đầu ra 2 chiều của ResNet50
x = Dense(1024, activation='relu')(x)  # Lớp FC đầu tiên
predictions = Dense(36, activation='softmax')(x)  # Lớp dự đoán (36 lớp)

# Xây dựng mô hình đầy đủ
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze các lớp của ResNet50 để không bị huấn luyện lại
for layer in base_model.layers:
    layer.trainable = False

# Biên dịch mô hình
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Thêm EarlyStopping để dừng huấn luyện sớm nếu không có cải thiện
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Huấn luyện mô hình
history = model.fit(
    X_train,
    Y_train,
    epochs=30,
    validation_data=(X_test, Y_test),
    callbacks=[early_stopping]
)

# Lưu mô hình đã huấn luyện
model.save('../Scripts/hand_sign_recognition_ResNet-50.h5')
