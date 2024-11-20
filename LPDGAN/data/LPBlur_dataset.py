import os
import cv2
from tqdm import tqdm
from .aug import L_CLAHE, normalize_brightness
from pathlib import Path
from paddleocr import PaddleOCR
import logging
from paddleocr.ppocr.utils.logging import get_logger
_paddle_logger = get_logger()
_paddle_logger.setLevel(logging.ERROR)
from typing import Optional, Literal
from torch.utils.data import Dataset
import torch
import json
import easyocr
import torch.nn as nn
import numpy as np
from .sp import Spatial_Pyramid_cv2

__all__ = ["LP_Deblur_Inference_Dataset", "LP_Deblur_OCR_Valiation_Dataset", "LP_Deblur_Dataset"]

def get_easy_ocr_rcnn():
    reader = easyocr.Reader(['en'], gpu=False)  # You can set gpu=True if you have a GPU
    # Access the recognition model (CRNN)
    fe = reader.recognizer
    for param in fe.parameters():
        param.requires_grad = False
    fe.eval()
    return fe  


flatten2D = lambda  nested_list: [item for sublist in nested_list for item in sublist]

class LP_Deblur_Inference_Dataset(Dataset):
    
    def __init__(self, imgs:list[Path], org_size:tuple[int,int]=(224,112), on_brightness:Optional[int]=180):
        super().__init__()
        self.sp = Spatial_Pyramid_cv2(org_size=org_size, origin_brightness=on_brightness)
        self.imgs = imgs
    
    def __len__(self) -> int:
        return len(self.imgs)
    
    def __getitem__(self, index) -> dict[str, torch.Tensor|str]:
        blur_img = cv2.imread(self.imgs[index]) 
        r = self.sp(img=blur_img, L=3, map_key="A")
        r['path'] = str(self.imgs[index])
        return r

class LP_Deblur_OCR_Valiation_Dataset(LP_Deblur_Inference_Dataset):
    
    def __init__(self, imgs:list[Path], labels:list[str], org_size = (224, 112), on_brightness = 180):
        super().__init__(imgs, org_size, on_brightness)
        self.labels = labels
        assert len(labels) == len(self.imgs)

    def __getitem__(self, index) -> dict[str, torch.Tensor|str]:
        sp_img = super().__getitem__(index)
        sp_img['gth'] = self.labels[index]
        return sp_img
    
    @classmethod
    def build_dataset(cls, dataroot:Path, label_file:os.PathLike, org_size = (224, 112), on_brightness = 180) -> "LP_Deblur_OCR_Valiation_Dataset":
        
        if label_file is None or dataroot is None:
            return None
        
        if not dataroot.is_dir():
            return None
        
        if not os.path.exists(str(label_file)):
            return None
        
        l:dict[str, str] = None
        with open(label_file, "r") as f:
            l= json.load(f)
        imgs = [dataroot/i for i in l.keys()]
        for i in imgs:
            assert i.is_file()
        
        labels = list(l.values())
        return cls(imgs=imgs, labels=labels, org_size=org_size, on_brightness=on_brightness)

class LP_Deblur_Dataset(Dataset):
    
    def __init__(self, data_root:Path, blur_aug:list[str], mode:Literal['train', 'test']="train", org_size:tuple[int, int]= (224, 112), on_brightness:Optional[int]=None) -> None:
        
        super().__init__()
        self.mode = mode
        self.org_size = org_size
        
        self.ocr = PaddleOCR(use_angle_cls=True, lang="en")
 
        self.blur_aug = blur_aug
        self.sharp_root = Path(data_root)/"sharp"
        self.txt_info = {}
        for imgid in tqdm([_.name for _ in (self.sharp_root).glob("*.jpg")]):
            txt_tensor = self.get_text_info(img=self.sharp_root/imgid) 
            if len(txt_tensor):
                self.txt_info[imgid] = txt_tensor
      
        self.sharp_blur_pairs: list[tuple[Path, Path]] = flatten2D([
            [
                (self.sharp_root/f"{imgid}", Path(data_root)/f"{b}"/f"{imgid}") 
                for imgid in self.txt_info.keys()
            ] 
            for b in self.blur_aug
        ])
        
        for t in self.sharp_blur_pairs:
            assert t[1].is_file(), print(t[1])
            if t[0] is not None:
                assert t[0].is_file()
                assert int(t[1].stem) == int(t[0].stem) 
        
        self.N_pairs = len(self.sharp_blur_pairs)
        self.sp = Spatial_Pyramid_cv2(org_size=self.org_size, origin_brightness=on_brightness)


    def __len__(self)->int:
        return self.N_pairs
    
    def __getitem__(self, idx) -> dict[str, torch.Tensor|str]:
        ps = self.sharp_blur_pairs[idx]

        blur_path = ps[1]
        blur_img = cv2.imread(blur_path)  
        r = self.sp(img=blur_img, L=3, map_key="A")
        r['A_paths'] = str(blur_path)
        sharp_image = cv2.imread(ps[0])
        r['plate_info'] = self.txt_info[ps[0].name]
        r = r | self.sp(img=sharp_image, L=4, map_key="B")
        r['B_paths']= str(ps[1])
        
        return r
    
    def get_text_info(self, img:Path) -> torch.Tensor:
             
        def count_area(d)->int:    
            if d is None:
                return 0
            return (d[0][1][0] - d[0][0][0])*(d[0][2][1] - d[0][0][1])
        
        i = L_CLAHE(normalize_brightness(cv2.resize(cv2.imread(img), self.org_size)))
        result = self.ocr.ocr(i, cls=True)[0]
        
        if result is None:
            return []
        
        main_patch = np.argmax(np.array([count_area(r) for r in result]))
        t= torch.from_numpy(result[main_patch][1][2])
        if len(t) < 10:
            t = torch.cat([t, torch.zeros(10-len(t))])
        elif len(t)>10:
            raise ValueError(f"{img} in paddle OCR get too long region")
        return t
