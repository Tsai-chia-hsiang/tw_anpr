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
