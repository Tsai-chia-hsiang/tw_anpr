from PIL import Image
from pathlib import Path
import numpy as np
import cv2
import torch
from torchvision import transforms


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


class MultiResolutionImage():
    
    def __init__(self, org_size:tuple[int, int]=(112, 224)) -> None:
        self.PP=[
            transforms.Compose(
                [transforms.Resize(size=org_size),
                transforms.ToTensor(),
                transforms.Normalize(std=[0.5, 0.5, 0.5], mean=[0.5,0.5,0.5])]
            ),
            transforms.Compose(
                [transforms.Resize(size=(org_size[0]//2, org_size[1]//2)),
                transforms.ToTensor(),
                transforms.Normalize(std=[0.5, 0.5, 0.5], mean=[0.5,0.5,0.5])]
            ),
            transforms.Compose(
                [transforms.Resize(size=(org_size[0]//4, org_size[1]//4)),
                transforms.ToTensor(),
                transforms.Normalize(std=[0.5, 0.5, 0.5], mean=[0.5,0.5,0.5])]
            ),
            transforms.Compose(
                [transforms.Resize(size=(org_size[0]//8, org_size[1]//8)),
                transforms.ToTensor(),
                transforms.Normalize(std=[0.5, 0.5, 0.5], mean=[0.5,0.5,0.5])]
            )
        ]
    def __call__(self, img:str|Path|np.ndarray, map_key:str="A", L:int=4, to_batch:bool=False) -> dict[str, torch.Tensor]:
        
        def a_res(img:Image.Image, L_idx:int, to_batch:bool=False) -> torch.Tensor:
            i:torch.Tensor = self.PP[L_idx](img)
            if to_batch:
                i = i.unsqueeze(0)
            return i  
        
        img_ = img
        if isinstance(img_, (str, Path)):
            img_ = cv2.imread(img)
        img_ = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_ = Image.fromarray(img_)
        assert L < 5 and L > 1
        return { f"{map_key}{li}" : a_res(img_, li, to_batch=to_batch) for li in range(L)}

