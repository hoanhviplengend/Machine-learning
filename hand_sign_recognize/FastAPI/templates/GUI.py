import tkinter as tk
import numpy as np
import cv2
from PIL import Image, ImageTk

# Tạo cửa sổ chính
root = tk.Tk()

# Thiết lập tiêu đề và kích thước cho cửa sổ
root.title("Ứng dụng nhận diện cử chỉ tay")
root.geometry("1920x1080")  # Chiều rộng x Chiều cao

# Thêm nhãn vào cửa sổ
label = tk.Label(root, text="Mô hình nhận diện ngôn ngữ kí hiệu tay!", font=("Arial", 50, "bold"))
label.pack(pady=50)  # padding theo chiều dọc

# Thêm một nút bấm
button = tk.Button(root, text="Nhấn vào đây", font=("Arial", 14), command=lambda: print("Nút đã được nhấn"))
button.pack(pady=10)
# Tạo Canvas
canvas = tk.Canvas(root, width=500, height=500)
canvas.pack()

# Tải ảnh bằng Pillow

# Đọc ảnh bằng OpenCV

img = Image.open(r"/hand_sign_recognize/Data/Gesture Image Pre-Processed Data - Test/0/769.jpg")

# Vẽ hình chữ nhật lên ảnh
image = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGBA2BGR)
cv2.rectangle(image, (100, 100), (200, 200), (0, 0, 255), 2)

# Chuyển đổi ảnh từ BGR (OpenCV) sang RGB
img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Chuyển đổi ảnh từ NumPy array sang định dạng mà Tkinter hiểu
img_pil = Image.fromarray(image)
img_tk = ImageTk.PhotoImage(image=img_pil)

# Tạo Canvas và hiển thị ảnh trên Canvas
canvas = tk.Canvas(root, width=image.shape[1], height=image.shape[0])
canvas.pack()
canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
root.mainloop()
