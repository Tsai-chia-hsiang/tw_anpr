from pathlib import Path
import numpy as np
import cv2
import torch

def tensor2img(input_image:torch.Tensor, imtype=np.uint8, to_cv2:bool=True)->np.ndarray:
        
    """
    Converts a Tensor array into a numpy image array.
    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
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

def load_networks(net:torch.nn.Module, pretrained_ckpt:Path):
    
    if pretrained_ckpt is None:
        print(f"{pretrained_ckpt} is None, will not load")
        return 
    
    if not pretrained_ckpt.is_file():
        print(f"No such {pretrained_ckpt} file, will not load")
        return 
    
    print(f"loading the model from {pretrained_ckpt}")
    state_dict = torch.load(pretrained_ckpt, map_location='cpu', weights_only=True)
    
    if hasattr(state_dict, '_metadata'):
        del state_dict._metadata
    
    net.load_state_dict(state_dict)

