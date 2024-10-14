import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model  # type: ignore

# Khởi tạo các đối tượng từ MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
# Tải mô hình
model = load_model(r'..\Scripts\best_model_lenet.h5')

# Danh sách tên lớp
class_names = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y'
]


# Hàm chuẩn bị ảnh
def prepare_image(image):
    # image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Không thể đọc ảnh")

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(gray_image, (28, 28))
    normalized_image = resized_image / 255.0
    processed_image = np.stack((normalized_image,) * 3, axis=-1)
    processed_image = np.expand_dims(processed_image, axis=0)

    return processed_image


# Hàm dự đoán lớp của ảnh
def predict_class(image):
    prediction = model.predict(image)
    prediction_class_index = np.argmax(prediction, axis=-1)[0]
    confidence = prediction[0][prediction_class_index]

    if confidence < 0.5:  # Ngưỡng cho độ tin cậy
        return "Không chắc chắn"  # Không chắc chắn về dự đoán
    return class_names[prediction_class_index]


# Khởi tạo nhận diện bàn tay
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)


# Xóa nền
def remove_background(image):
    # Chuyển đổi ảnh sang không gian màu HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Định nghĩa khoảng màu cho nền (có thể điều chỉnh)
    lower_background = np.array([0, 0, 0])  # Thay đổi giá trị cho phù hợp
    upper_background = np.array([0, 0, 0])  # Thay đổi giá trị cho phù hợp

    # Tạo mặt nạ cho nền
    mask = cv2.inRange(hsv, lower_background, upper_background)

    # Lấy vùng bàn tay
    mask_inv = cv2.bitwise_not(mask)
    hand = cv2.bitwise_and(image, image, mask=mask_inv)

    return hand


def run():
    # Khởi tạo camera
    cap = cv2.VideoCapture(0)

    # Biến để đếm số ảnh chụp
    image_counter = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Không thể truy cập camera.")
            break

        # Lật khung hình để cảm giác giống gương
        frame = cv2.flip(frame, 1)

        # Chuyển đổi BGR sang RGB vì MediaPipe hoạt động với RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Xử lý khung hình và nhận diện bàn tay
        results = hands.process(rgb_frame)

        # Nếu tìm thấy bàn tay
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Vẽ các điểm mốc (landmarks) trên bàn tay
                # mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Tính toán vị trí và kích thước của khung hình 64x64
                # vị trí số 9 trên bàn tay
                height, width, _ = frame.shape
                x_center = int(hand_landmarks.landmark[9].x * width)
                y_center = int(hand_landmarks.landmark[9].y * height)

                # Tính toán tọa độ cho khung 64x64
                x_start = max(0, x_center - 170)
                y_start = max(0, y_center - 170)
                x_end = min(width, x_center + 170)
                y_end = min(height, y_center + 170)

                # Vẽ khung 64x64
                cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)

                # Hiển thị khung hình
                cv2.imshow('MediaPipe Hands', frame)

                # Nhấn 'c' để chụp ảnh
                cropped_hand = frame[y_start:y_end, x_start:x_end]
                if cropped_hand.size > 0:
                    image = remove_background(cropped_hand)
                    prepare_img = prepare_image(image)
                    predicted_class = predict_class(prepare_img)
                    # In ra kết quả dự đoán
                    print(f'Lớp dự đoán: {predicted_class}')
                    image_counter += 1
                    # Chuyển đổi lại prepare_img về định dạng có thể hiển thị
                    display_img = (prepare_img[0] * 255).astype(np.uint8)  # Chuyển đổi về định dạng 0-255

                    # Thay đổi kích thước ảnh (nếu cần)
                    display_img_resized = cv2.resize(display_img, (128, 128))  # Thay đổi kích thước nếu cần thiết

                    # Tạo frame hiển thị ảnh đã qua xử lý
                    cv2.imshow("Ảnh đã qua xử lý", display_img_resized)
        # Nhấn 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Giải phóng tài nguyên
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
