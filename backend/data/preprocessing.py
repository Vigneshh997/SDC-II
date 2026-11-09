"""PPT image enhancement followed by the pretrained encoder's own transforms."""

import numpy as np
from PIL import Image


class ImagePreprocessing:
    def __init__(self, pretrained_transform, clahe=False):
        self.pretrained_transform, self.clahe = pretrained_transform, clahe

    def __call__(self, image):
        if self.clahe:
            import cv2

            gray = np.asarray(image.convert("L"))
            enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
            image = Image.fromarray(enhanced).convert("RGB")
        return self.pretrained_transform(image)
