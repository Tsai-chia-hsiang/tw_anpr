import cv2
from tqdm import tqdm
import gc
import json
import numpy as np
from pathlib import Path

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

def moire_vertical(image, cycle_angle=60, blending_alpha:float=0.15):

    # Get the dimensions of the image
    rows, cols, _ = image.shape

    # Generate a vertical sinusoidal moiré pattern
    x = np.arange(0, cols)
    X = np.tile(x, (rows, 1))  # Repeat the x values across all rows
    moire_pattern = 128 + 127 * np.sin(2 * np.pi * X / cycle_angle)

    # Normalize the moiré pattern to 0-255 and convert to uint8
    moire_pattern = np.clip(moire_pattern, 0, 255).astype(np.uint8)

    # Convert the grayscale moiré pattern to BGR by stacking the grayscale channel
    moire_pattern_bgr = cv2.merge([moire_pattern, moire_pattern, moire_pattern])

    # Blend the moiré pattern with the original BGR image
    blended_image = cv2.addWeighted(image, 1 - blending_alpha, moire_pattern_bgr, blending_alpha, 0)

    return blended_image
  
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

def sharpening(img:np.ndarray)->np.ndarray:
    return cv2.filter2D(
        img, -1,  
        np.array(
            [[0, -1, 0],
            [-1,  5, -1],
            [0, -1, 0]])
    )


# ========================== differen level blur ======================#
def blur(src):
    aug = mosaic(src, factor=0.17)
    aug = moire_vertical(aug, cycle_angle=60, blending_alpha=0.1)
    aug = patches_blur(aug, psize=(28, 32), blur_n=14, gblur_para=(9, 10))
    aug = sharpening(aug)
    aug = patch_mosaic(aug, psize=(112, 16), blur_n=10, factor=0.2)
    aug = gblur(aug, ksize=5, sigma=5)
    return aug

def blur_median(src):
    aug = mosaic(src, factor=0.19)
    aug = moire_vertical(aug, cycle_angle=60, blending_alpha=0.1)
    aug = sharpening(aug)
    aug = patch_mosaic(aug, psize=(112, 16), blur_n=7, factor=0.2)
    aug = gblur(aug, ksize=5, sigma=5)
    return aug

def blur_mosiac(src):
    aug = mosaic(src, factor=0.15)
    aug = sharpening(aug)  
    return aug

def blur_little(src):
    aug = mosaic(src, factor=0.3)
    aug = moire_vertical(aug, cycle_angle=60, blending_alpha=0.1)
    aug = patches_blur(aug, psize=(28, 32), blur_n=14, gblur_para=(9, 10))
    aug = sharpening(aug)
    aug = patch_mosaic(aug, psize=(112, 16), blur_n=7, factor=0.3)
    aug = gblur(aug, ksize=7, sigma=5)   
    return aug

if __name__ == "__main__":
    with open("./tw/old/img_ids.json", "r") as f:
        old = json.load(f)
    imgs = [_ for _ in Path("./tw/new/sharp").glob("*.jpg") if _.stem not in old]
    R = Path("./tw/new/")

    blur_dir =R/"blur"
    blur_dir.mkdir(parents=True, exist_ok=True)

    blur_little_dir = R/"blur_little"
    blur_little_dir.mkdir(parents=True, exist_ok=True)

    blur_median_dir = R/"blur_median"
    blur_median_dir.mkdir(parents=True, exist_ok=True)
    
    blur_mosiac_dir = R/"blur_mosiac"
    blur_mosiac_dir.mkdir(parents=True, exist_ok=True)

    
    for i_path in tqdm(imgs):
        iid = i_path.name
        img_i = cv2.imread(i_path)

        cv2.imwrite(blur_dir/iid, blur(src=img_i))

        cv2.imwrite(blur_little_dir/iid, blur_little(src=img_i))
 
        cv2.imwrite(blur_median_dir/iid, blur_median(src=img_i))

        cv2.imwrite(blur_mosiac_dir/iid, blur_mosiac(src=img_i))