import cv2
import numpy as np

def detect_scalebar(image, threshold=190):
    h = image.shape[0]
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.dtype != 'uint8':
        image = cv2.convertScaleAbs(image, alpha=(255.0 / image.max()))
    cropped = image[int(h * 0.9):, :] 
    _, binary = cv2.threshold(cropped, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_width = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > max_width:
            max_width = w

    return max_width