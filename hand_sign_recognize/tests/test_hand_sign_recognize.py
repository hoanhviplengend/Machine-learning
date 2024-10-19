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


