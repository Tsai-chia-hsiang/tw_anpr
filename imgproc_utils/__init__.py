import math
import numpy as np
import cv2
from deskew import determine_skew
from .dehaze import image_dehazer

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def L_CLAHE(img:np.ndarray) -> np.ndarray:
    """
    Contrast Limited Adaptive Histogram Equalization at L channel 
    for L,a,b
    
    Will convert back to bgr then return
    """
    
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    l,a,b = cv2.split(img_lab)   
    l_clahe = clahe.apply(l)
    img_lab_clahe = cv2.merge((l_clahe, a, b))
    # Convert the result back to BGR color space
    img_clahe_bgr = cv2.cvtColor(img_lab_clahe, cv2.COLOR_Lab2BGR)
    return img_clahe_bgr

def normalize_brightness(img:np.ndarray, target_brightness=180)->np.ndarray:
    # Convert image to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Split into H, S, V channels
    h, s, v = cv2.split(hsv)
    
    # Calculate the current average brightness (V channel)
    current_brightness = np.mean(v)
    
    # Scale the V channel to match the target brightness
    brightness_ratio = target_brightness / current_brightness
    v = np.clip(v * brightness_ratio, 0, 255).astype(np.uint8)
    
    # Merge H, S, adjusted V channels back and convert to BGR
    hsv_normalized = cv2.merge([h, s, v])
    normalized_image = cv2.cvtColor(hsv_normalized, cv2.COLOR_HSV2BGR)
    
    return normalized_image

def deskew_plate(img:np.ndarray)->np.ndarray:
    
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    angle = determine_skew(grayscale)
    old_width, old_height = img.shape[:2]
    angle_radian = math.radians(angle)
    width = abs(np.sin(angle_radian) * old_height) + abs(np.cos(angle_radian) * old_width)
    height = abs(np.sin(angle_radian) * old_width) + abs(np.cos(angle_radian) * old_height)
    image_center = tuple(np.array(img.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rot_mat[1, 2] += (width - old_width) / 2
    rot_mat[0, 2] += (height - old_height) / 2
    return cv2.warpAffine(
        img, rot_mat, 
        (int(round(height)), int(round(width))), 
        borderValue=(0, 0, 0)
    )