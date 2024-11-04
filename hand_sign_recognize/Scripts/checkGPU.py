
import tensorflow as tf
import sys

def check_GPU():
    if len(tf.config.experimental.list_physical_devices('GPU')) > 0:
        print("Có GPU sẵn sàng.")
    else:
        print("Không tìm thấy GPU.")
        sys.exit()
