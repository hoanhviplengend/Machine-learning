
import cv2


threshold = 60
blurValue = 41
# Hàm chuẩn bị ảnh cho mô hình
def prepare_image(image):
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
