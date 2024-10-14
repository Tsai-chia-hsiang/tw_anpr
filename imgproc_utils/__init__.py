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

def sharpening(img:np.ndarray)->np.ndarray:
    return cv2.filter2D(
        img, -1,  
        np.array(
            [[0, -1, 0],
            [-1,  5, -1],
            [0, -1, 0]])
    )

def foggy(image:np.ndarray) -> np.ndarray:
    # Get the dimensions of the image
    height, width, _ = image.shape

    # Create random fog-like noise
    fog_noise = np.random.normal(loc=200, scale=30, size=(height, width, 3))  # Bright fog
    fog_noise = fog_noise.astype(np.uint8)

    # Blend the original image with the fog noise using weighted sum
    return cv2.addWeighted(image, 0.8, fog_noise, 0.2, 0)  # You can adjust the weights

def mosaic(img:np.ndarray, factor:float=0.2) -> np.ndarray:
    origin_size = img.shape[:2]
    return cv2.resize(
            cv2.resize(
                img, 
                (int(origin_size[1]*factor), int(origin_size[0]*factor))
            ),
            (origin_size[1], origin_size[0])
        )

def gblur(img:np.ndarray, ksize:int=7, sigma:float=7.0) -> np.ndarray:
    return cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)

def erosion_on_text(img: np.ndarray) -> np.ndarray:

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply a binary threshold to extract text
    _, binary_image = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    # Define the kernel for erosion
    kernel = np.ones((3, 3), np.uint8)

    # Apply erosion on the binary image
    eroded_binary = cv2.erode(binary_image, kernel, iterations=1)

    # Invert the binary image back to match the original BGR background
    eroded_binary_inv = cv2.bitwise_not(eroded_binary)

    # Create a 3-channel mask by repeating the grayscale mask for each color channel
    mask_3channel = cv2.merge([eroded_binary_inv] * 3)

    # Apply the mask on the original BGR image
    eroded_colored_image = cv2.bitwise_and(img, mask_3channel)

    return eroded_colored_image

def moire_vertical(img, cycle_angle=60, blending_alpha:float=0.15):

    # Get the dimensions of the image
    rows, cols, _ = img.shape

    # Generate a vertical sinusoidal moiré pattern
    x = np.arange(0, cols)
    X = np.tile(x, (rows, 1))  # Repeat the x values across all rows
    moire_pattern = 128 + 127 * np.sin(2 * np.pi * X / cycle_angle)

    # Normalize the moiré pattern to 0-255 and convert to uint8
    moire_pattern = np.clip(moire_pattern, 0, 255).astype(np.uint8)

    # Convert the grayscale moiré pattern to BGR by stacking the grayscale channel
    moire_pattern_bgr = cv2.merge([moire_pattern, moire_pattern, moire_pattern])

    # Blend the moiré pattern with the original BGR image
    blended_image = cv2.addWeighted(img, 1 - blending_alpha, moire_pattern_bgr, blending_alpha, 0)

    return blended_image

def simi_patches(img:np.ndarray, psize:tuple[int,int]=(28,32))->None:
    r = img.shape[0]//psize[0]
    c = img.shape[1]//psize[1]
    
    for ri in range(1, r):
        cv2.line(img, (0, psize[0]*ri), (img.shape[1] -1, psize[0]*ri), color=(0,0,255), thickness=1)

    for ci in range(1, c):
        cv2.line(img, (psize[1]*ci, 0), (psize[1]*ci, img.shape[0] -1), color=(0,0,255), thickness=1)
   
def patches_blur(
    img:np.ndarray, psize:tuple[int,int]=(28,32), 
    blur_n:int=5, inplace:bool=False, gblur_para:tuple[int, float]=(11, 10)
)->np.ndarray|None:
    
    image = img.copy() if not inplace else img
    r = img.shape[0]//psize[0]
    c = img.shape[1]//psize[1]
    ps = np.random.choice(np.arange(c, c*(r-1)), size=blur_n, replace=False)
    for p in ps:
        x, y =p//c, p%c # top left
        image[x*psize[0]:(x + 1)*psize[0], y*psize[1]:(y+1)*psize[1]] = gblur(
            image[x*psize[0]:(x + 1)*psize[0], y*psize[1]:(y+1)*psize[1]], 
            ksize=gblur_para[0], sigma=gblur_para[1]
        )
        
    if not inplace:
        return image

def patch_mosaic(
    img:np.ndarray, psize:tuple[int,int]=(28,32), 
    blur_n:int=5, inplace:bool=False, factor:float=0.2
):
    image = img.copy() if not inplace else img
    r = img.shape[0]//psize[0]
    c = img.shape[1]//psize[1]
    ps = np.random.choice(np.arange(0, c*r), size=blur_n, replace=False)
    for p in ps:
        x, y =p//c, p%c # top left
        image[x*psize[0]:(x + 1)*psize[0], y*psize[1]:(y+1)*psize[1]] = mosaic(
            image[x*psize[0]:(x + 1)*psize[0], y*psize[1]:(y+1)*psize[1]], 
            factor=factor
        )
        
    if not inplace:
        return image
  
def motion_blur(image:np.ndarray, size:int, angle) -> np.ndarray:
    k = np.zeros((size, size), dtype=np.float32)
    k[ (size-1)// 2 , :] = np.ones(size, dtype=np.float32)
    k = cv2.warpAffine(k, cv2.getRotationMatrix2D( (size / 2 -0.5 , size / 2 -0.5 ) , angle, 1.0), (size, size) )  
    k = k * ( 1.0 / np.sum(k) )        
    return cv2.filter2D(image, -1, k) 

if __name__ == "__main__":
    src = cv2.imread("../dataset/tw/train/sharp/0.jpg")
    aug = normalize_brightness(img=src)
    #aug = motion_blur(aug, 10, 15)
    #aug = foggy(image=aug)
    aug = mosaic(aug, factor=0.17)
    aug = moire_vertical(aug, cycle_angle=60, blending_alpha=0.1)
    aug = patches_blur(aug, psize=(28, 32), blur_n=14, gblur_para=(9, 10))
    aug = sharpening(aug)
    aug = patch_mosaic(aug, psize=(112, 16), blur_n=10, factor=0.13)
    aug = gblur(aug, ksize=5, sigma=5)
