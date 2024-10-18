
import cv2
import numpy as np
import os
threshold = 60
blurValue = 41
# Danh sách tên lớp
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
               'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
               'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']


# Hàm chuẩn bị ảnh cho mô hình
def prepare_image(image):
    # image=cv2.resize(image,(32,32))
    # image=image.reshape(1,32,32,3)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)

    ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return ret, thresh


def remove_background(image):
    # Chuyển đổi ảnh sang không gian màu xám
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Áp dụng GaussianBlur để làm mịn ảnh
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Áp dụng ngưỡng để tạo mặt nạ nhị phân (binary mask)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Tạo một mặt nạ ngược (inverse mask)
    mask_inv = cv2.bitwise_not(mask)

    # Tạo ảnh mới với nền đã được xóa
    background_removed = cv2.bitwise_and(image, image, mask=mask_inv)

    return background_removed

# Dự đoán ảnh
def predict_class(model, image):
    prediction = model.predict(image)
    prediction_class_index = np.argmax(prediction, axis=-1)[0]
    confidence = prediction[0][prediction_class_index]

    if confidence < 0.5:  # Ngưỡng cho độ tin cậy
        return "Không chắc chắn"  # Không chắc chắn về dự đoán
    return class_names[prediction_class_index]

# Lật ảnh theo chiều ngang
def flip_images_horizontally(folder_path):
    # Duyệt qua các file trong folder_path
    for file in os.listdir(folder_path):
        if file.endswith((".jpg", ".png", ".jpeg")):  # Chỉ lấy các file ảnh
            img_path = os.path.join(folder_path, file)

            # Đọc ảnh bằng OpenCV
            img = cv2.imread(img_path)

            # Kiểm tra xem ảnh có được đọc thành công không
            if img is None:
                print(f"Không thể mở được ảnh {img_path}")
                continue

            # Lật ảnh theo chiều ngang
            flipped_img = cv2.flip(img, 1)

            # Tạo tên file mới cho ảnh lật (thêm tiền tố 'flipped_')
            #output_path = os.path.join(folder_path, "flipped_" + file)

            # Lưu ảnh đã lật
            cv2.imwrite(img_path, flipped_img)

    print("Đã lật tất cả ảnh theo chiều ngang.")
