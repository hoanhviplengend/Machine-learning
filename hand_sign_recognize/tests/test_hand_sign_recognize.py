#!/usr/bin/env python

"""Tests for `hand_sign_recognize` package."""

import unittest
import cv2
import numpy as np
from rembg import remove
import matplotlib.pyplot as plt
import os
from PIL.Image import Image
from tensorboard.plugins.image.summary_v2 import image
from tensorflow.keras.models import load_model
from hand_sign_recognize.src import image_processing
from hand_sign_recognize.src.image_processing import remove_background


class TestHand_sign_recognize(unittest.TestCase):
    """Tests for `hand_sign_recognize` package."""

    def setUp(self):
        """Set up test fixtures, if any."""

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_000_something(self):
        """Test something."""




# image = cv2.imread("o.jpg")
# if image is None:
#     print("Lỗi: Không thể đọc tệp ảnh. Vui lòng kiểm tra đường dẫn hoặc tệp.")
# image_rmb = remove(image)
# ret, image_prepare = image_processing.prepare_image(image_rmb)
# plt.imshow(cv2.cvtColor(cv2.resize(image_prepare, None, fx=0.5, fy=0.5), cv2.COLOR_BGR2RGB))
# plt.title("After Preprocessing")
# plt.axis('off')  # Tắt trục
# plt.show()
# model = load_model("../Scripts/best_model_lenet.h5")
# img = np.stack((image_prepare, image_prepare, image_prepare), axis=-1)
# image_resize = cv2.resize(img, (32, 32))
# image_resize = image_resize.reshape(1, 32, 32, 3)
# print(image_processing.predict_class(model, image_resize))
