import time
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from image_processing import remove_background,prepare_image
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Tải mô hình


# Danh sách tên lớp
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
               'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
               'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# Hàm dự đoán lớp của ảnh
def predict_class(model,image):
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

# Biến để đếm số ảnh chụp
image_counter = 0
cap_region_x_begin = 0.5
cap_region_y_end = 0.8


bgSubThreshold = 50
learningRate = 1
predThreshold = 95
isBgCaptured = 0

# Khởi tạo camera
cap = cv2.VideoCapture(0)
cap.set(10, 200)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)  # Đặt độ phân giải khung hình


def run(model_path):
    model = load_model(model_path)
    if model:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Không thể đọc khung hình từ camera")
                break

            # Làm mịn ảnh
            frame = cv2.bilateralFilter(frame, 5, 50, 100)
            # Lật khung hình để cảm giác giống gương
            frame = cv2.flip(frame, 1)
            # Vẽ khung hình chữ nhật
            cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),
                          (frame.shape[1] - 50, int(cap_region_y_end * frame.shape[0]) - 100), (255, 0, 0), 2)

            # Xóa nền ảnh
            image = remove_background(frame)
            cv2.imshow("Background Removed", image[0:int(cap_region_y_end * frame.shape[0]) - 100,
                            int(cap_region_x_begin * frame.shape[1]):frame.shape[1] - 50])

            # Cắt ảnh theo vùng đã định nghĩa
            cropped_image = image[0:int(cap_region_y_end * frame.shape[0]) - 100,
                            int(cap_region_x_begin * frame.shape[1]):frame.shape[1] - 50]

            ret, thresh = prepare_image(cropped_image)
            cv2.imshow("Thresh", cv2.resize(thresh, None, fx=0.5, fy=0.5))

            if np.count_nonzero(thresh) / (thresh.shape[0] * thresh.shape[1]) > 0.2:
                # Chuyển đổi định dạng hình ảnh cho dự đoán
                # thresh = cv2.imread(r"..\Hand-Sign-Recognition\Gesture Image Pre-Processed Data - Test\C\1.jpg")
                # thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
                # target = np.stack((thresh, thresh, thresh), axis=-1)  # Tạo ảnh 3 kênh từ ảnh xám
                # target = cv2.resize(target, (32, 32))
                # target = target.reshape(1, 32, 32, 3)
                target = np.stack((thresh, thresh, thresh), axis=-1)
                target = cv2.resize(target, (32, 32))
                target = target.reshape(1, 32, 32, 3)
                result = predict_class(model,target)
                print("Dự đoán:", result)

            # Hiển thị khung hình gốc
            cv2.imshow("Original", cv2.resize(frame, None, fx=0.5, fy=0.5))
            # cv2.putText(frame, "Background captured", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 10,
            #             lineType=cv2.LINE_AA)
            # time.sleep(2)
            k = cv2.waitKey(10)
            if k == ord("q"):
                break
            elif k == ord('k'):
                cv2.putText(frame, "Background reset", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 10,
                            lineType=cv2.LINE_AA)
                time.sleep(1)

        # Giải phóng tài nguyên
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
