import numpy as np
import cv2

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