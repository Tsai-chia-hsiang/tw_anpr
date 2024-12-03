from typing import Optional
import numpy as np
import cv2
import torch
from torchvision import transforms
from .aug import normalize_brightness

class Spatial_Pyramid_cv2():
    
    def __init__(self, org_size:tuple[int, int]=(112, 56), origin_brightness:Optional[int]=180, interpolation:int=cv2.INTER_CUBIC) -> None:
        """
        Args:
        -- 
        - org_size : origin input size :
            - [W, H]
        
        """
        
        self.b = origin_brightness
        self.inter = interpolation
        self.tsize = [(org_size[0]//(2**L_idx), org_size[1]//(2**L_idx)) for L_idx in range(5)]
        self.n = transforms.Compose(
            [
                transforms.ToTensor(), 
                transforms.Normalize(std=[0.5, 0.5, 0.5], mean=[0.5,0.5,0.5])
            ]
        )
    
  
    def __call__(self, img:np.ndarray, map_key:str="A", L:int=2, to_batch:bool=False) -> dict[str, torch.Tensor]:
        
        def a_res(img, L_idx:int, to_batch:bool=False) -> torch.Tensor:
            x = cv2.resize(img, self.tsize[L_idx], interpolation=self.inter) if L_idx > 0 else img
            i:torch.Tensor = self.n(x)
            return i.unsqueeze(0) if (to_batch and i.ndim == 3) else i
        
        img_ = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), self.tsize[0], interpolation=self.inter) 
        if self.b is not None:
            img_ = normalize_brightness(img=img_, target_brightness=self.b)
        
        assert L < 5 and L > 1
        return { f"{map_key}{li}" : a_res(img_, li, to_batch=to_batch) for li in range(L)} 



"""
PIL framework:
SP = [
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
"""