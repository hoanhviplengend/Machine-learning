from vit_keras import vit
from tensorflow.keras.callbacks import EarlyStopping
from hand_sign_recognize.Scripts.splitData import X_train, X_test, Y_train, Y_test
from hand_sign_recognize.Scripts.checkGPU import check_GPU

# Kiểm tra GPU
check_GPU()

# Tạo mô hình ViT
model = vit.vit_b16(
    image_size=32,
    activation='softmax',
    pretrained=True,
    include_top=True,
    pretrained_top=False,
    classes=36
)

# Biên dịch mô hình
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Thêm EarlyStopping để dừng huấn luyện sớm nếu không có cải thiện
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Huấn luyện mô hình
history = model.fit(
    X_train,
    Y_train,
    epochs=20,
    validation_data=(X_test, Y_test),
    callbacks=[early_stopping]  # Thêm EarlyStopping vào callback
)

# Lưu mô hình đã huấn luyện
model.save('../Scripts/hand_sign_recognition_ViT.h5')
