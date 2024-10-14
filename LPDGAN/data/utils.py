import numpy as np
import cv2
import torch

def tensor2img(input_image:torch.Tensor, imtype:np.dtype=np.uint8, to_cv2:bool=True)->np.ndarray:  
    """
    Converts a Tensor array into a numpy image array.
    Args:
    -- 
    - input_image (torch.tensor):  
        - the input image tensor array
    - imtype (np.dtype) : 
        - the desired type of the converted numpy array, default is np.uint8
    - to_cv2 (boolean):
        - if True, will convert RGB 2 BGR
    """
    
    image_tensor = input_image.data
    # debatch & convert it into a numpy array
    image_numpy = image_tensor[0].cpu().float().numpy()  
    if image_numpy.shape[0] == 1:  # grayscale to RGB
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    
    # CxHxW -> HxWXC
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    image_numpy = image_numpy.astype(imtype)
    if to_cv2:
        image_numpy = cv2.cvtColor(image_numpy, cv2.COLOR_RGB2BGR)

    return image_numpy