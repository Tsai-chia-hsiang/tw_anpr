from torch.utils.data import Dataset
from pathlib import Path
from torchvision import transforms
from typing import Optional
import cv2
import torch
from tqdm import tqdm
import os
import numpy as np

from . import SR_FILE_DIR
import sys
anpr_path = Path(SR_FILE_DIR).parent/"anpr"
sys.path.append(os.path.abspath(anpr_path))
from anpr import LicensePlate_OCR

import json
def read_json(file):
    with open(file, "r") as f:
        return json.load(f)



class SR_Dataset(Dataset):
    
    lp_ocr=LicensePlate_OCR()
    
    def __init__(self, lr:Path, hr:Path, preload:bool=False, label_file:Optional[os.PathLike]=None, need_label:bool=True):
        super().__init__()
        self.blur_sharp_pairs = list(
            zip(
                sorted([_ for _ in lr.iterdir()]), 
                sorted([_ for _ in hr.iterdir()])
            )
        )
        
        self.label = None
        if need_label:
            self.label = self._paring_label(lf=label_file)
        
        for lr_i, hr_i in self.blur_sharp_pairs:
            assert lr_i.stem == hr_i.stem
        self._len = len(self.blur_sharp_pairs)

        self.to_tensor = transforms.Compose(
            [transforms.ToTensor()]
        )
        self.preload = preload
        self.blur_sharp_img_pairs = None
        if self.preload:
            self.preload_img()
    
    def _paring_label(self, lf:os.PathLike) -> dict[str, str]:
        
        need_ocr = False
        if lf is None:
            need_ocr=True
        else:
            if not os.path.exists(lf):
                need_ocr=True
        if need_ocr:
            return {
                j.stem: SR_Dataset.lp_ocr(cv2.imread(j))[1]
                for _, j in self.blur_sharp_pairs
            }
    
        return read_json(lf)
        
    def preload_img(self):
        self.blur_sharp_img_pairs = [
            (
                self.to_tensor( self.cv_read(i)),
                self.to_tensor( self.cv_read(j))
            ) 
            for i,j in tqdm(self.blur_sharp_pairs)
        ]

    def cv_read(self, p:os.PathLike)->np.ndarray:
        return cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
    
    def __len__(self):
        return self._len

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor, str, str, str]:
        
        bpath = str(self.blur_sharp_pairs[index][0])
        spath = self.blur_sharp_pairs[index][1]
        label = self.label[spath.stem]
        spath = str(spath)
        

        if self.preload:
            return (*self.blur_sharp_img_pairs[index], bpath, spath, label)
        
    
        return self.to_tensor(self.cv_read(bpath)), \
            self.to_tensor(self.cv_read(spath)), \
            bpath, spath, label

class SR_View_Cmp_Dataset(SR_Dataset):
    
    def __init__(self, lr, hr, preload = False):
        super().__init__(lr, hr, preload, need_label=False)
    
    def preload_img(self):
        self.blur_sharp_img_pairs = [
            (
                self.to_tensor( self.cv_read(i)),
                str(j)
            ) for i,j in tqdm(self.blur_sharp_pairs)
        ]
    def __getitem__(self, index) -> tuple[torch.Tensor, str]:
        
        if self.preload:
            return self.blur_sharp_img_pairs[index]
        
        bpath = self.blur_sharp_pairs[index][0]
        spath = self.blur_sharp_pairs[index][1]


        return self.to_tensor(self.cv_read(bpath)), str(spath)
