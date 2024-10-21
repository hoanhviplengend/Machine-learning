import time
import cv2
import mediapipe as mp
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.python.data.experimental.ops.testing import sleep

from image_processing import predict_image

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# khởi tạo khung hình ban đầu
cap_region_x_begin = 0.5
cap_region_y_end = 0.8

# Khởi tạo camera
cap = cv2.VideoCapture(0)
cap.set(10, 200)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)  # Đặt độ phân giải khung hình

# Khởi tạo nhận diện bàn tay
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2, # số tay có thể dùng
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)


# hàm dự đoán bằng ảnh
def predict_by_photo(model_path, image_path):
    try:
        model = load_model(model_path)
        if model:
            image = cv2.imread(image_path)
            # ngoại trừ "video" thì truyền type gì cũng được
            result, img_prepared = predict_image(model, image, "photo")
            cv2.imshow("Thresh", cv2.resize(img_prepared, (310,310), fx=0.5, fy=0.5))
            print("Dự đoán:", result)
            sleep(3)
            return True , result
    except Exception as e:
        print(e)
        return False , ""



# hàm dự đoán qua video camera
def predict_by_video(model_path, type):
    try :
        letters = list()
        model = load_model(model_path)
        photo_path = f'photo_taken.jpg'
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

                # Cắt ảnh theo vùng đã định nghĩa
                cropped_image = frame[0:int(cap_region_y_end * frame.shape[0]) - 100,
                                int(cap_region_x_begin * frame.shape[1]):frame.shape[1] - 50]

                # Hiển thị khung hình gốc
                cv2.imshow("Original", cv2.resize(frame, None, fx=0.5, fy=0.5))
                k = cv2.waitKey(10)

                if np.count_nonzero(cropped_image) / (cropped_image.shape[0] * cropped_image.shape[1]) > 0.2:
                    # dự đoán
                    if type == "video":
                        result, img_prepared = predict_image(model, cropped_image, type)
                        # Hiển thị khung hình đã xử lý
                        cv2.imshow("Thresh", cv2.resize(img_prepared, None, fx=0.5, fy=0.5))
                        print("Dự đoán:", result)
                        letters.append(result)
                    elif type == "photo":
                        # xóa đi ảnh đã chụp trước đó
                        if os.path.exists(photo_path):
                            os.remove(photo_path)
                    else:
                        print("Nhập lại type")
                        return

                if k == ord("q"):
                    # Giải phóng tài nguyên
                    cap.release()
                    cv2.destroyAllWindows()
                    hands.close()
                    return True, letters
                elif k == ord('c') and type != "video":  # Nhấn 'c' để chụp ảnh
                    cv2.imwrite(photo_path, cropped_image)  # Lưu ảnh
                    # thưực hiện dự đoán
                    letters.append(predict_by_photo(model_path, photo_path)[1])

                elif k == ord('k'):
                    cv2.putText(frame, "Background reset", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 10,
                                lineType=cv2.LINE_AA)
                    time.sleep(1)


    except Exception as e:
        print(e)
        return False, []
