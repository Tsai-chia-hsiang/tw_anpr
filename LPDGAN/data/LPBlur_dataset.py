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
import torch.nn as nn
import numpy as np
from .sp import Spatial_Pyramid_cv2
from .rec import EXT_MAP

__all__ = ["LP_Deblur_Inference_Dataset", "LP_Deblur_OCR_Valiation_Dataset", "LP_Deblur_Dataset"]

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
    
    def __init__(
        self, data_root:Path, blur_aug:list[str], 
        extraction:Literal['easyocr', 'paddleocr']='paddleocr',
        org_size:tuple[int, int]= (224, 112), 
        cached_file:Optional[Path]=None, cache_name:Optional[Path]=None,
        preload:bool=False
    ) -> None:
        
        super().__init__()
        self.org_size = org_size
        self.extraction = extraction
        self.blur_aug = blur_aug
        self.sharp_root = data_root/"sharp"
        
        self.txt_info = None
        ext_flag = False
        if cached_file is not None:
            if cached_file.is_file():
                self.txt_info = torch.load(cached_file)
            else:
                ext_flag=True
        else:
            ext_flag = True
        
        if ext_flag:
            self.txt_info = {}
            for imgid in tqdm([_.name for _ in (self.sharp_root).glob("*.jpg")]):
                txt_tensor = EXT_MAP[self.extraction](img=self.sharp_root/imgid, on_size=self.org_size) 
                if len(txt_tensor):
                    self.txt_info[imgid] = txt_tensor
            if cache_name is not None:
                if not cache_name.parent.is_dir():
                    cache_name.parent.mkdir(parents=True, exist_ok=True)
                torch.save(self.txt_info, cache_name)
       
        self.txt_d = tuple(
            self.txt_info[list(self.txt_info.keys())[0]].size()
        )
        
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
                assert t[1].stem == t[0].stem 
        
        self.N_pairs = len(self.sharp_blur_pairs)
        self.sp = Spatial_Pyramid_cv2(org_size=self.org_size)
        self.preload = preload
        self.buf = {}
        if self.preload:
            for i in tqdm(self.sharp_blur_pairs, desc='preloading src images'):
                if i[0] not in self.buf:
                    self.buf[i[0]] = self.sp(img=cv2.imread(i[0]), L=4, map_key="B")
                
                self.buf[i[1]] = self.sp(img=cv2.imread(i[1]), L=3, map_key="A")

    def __len__(self)->int:
        return self.N_pairs
    
    def __getitem__(self, idx) -> dict[str, torch.Tensor|str]:
        ps = self.sharp_blur_pairs[idx]
        r = None
        if not self.preload:
            r = self.sp(img=cv2.imread(ps[1]), L=3, map_key="A") | \
                self.sp(img=cv2.imread(ps[0]), L=4, map_key="B")
        else:
            r = self.buf[ps[1]]|self.buf[ps[0]]
        r['A_paths'] = str(ps[1])
        r['plate_info'] = self.txt_info[ps[0].name]
        return r
