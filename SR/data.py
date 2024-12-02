from torch.utils.data import Dataset
from pathlib import Path
from torchvision import transforms
import cv2
import torch
from tqdm import tqdm
import os
import numpy as np

class SR_Dataset(Dataset):
    
    def __init__(self, lr:Path, hr:Path, preload:bool=False):
        super().__init__()
        self.blur_sharp_pairs = list(
            zip(
                sorted([_ for _ in lr.iterdir()]), 
                sorted([_ for _ in hr.iterdir()])
            )
        )
        for lr_i, hr_i in self.blur_sharp_pairs:
            assert lr_i.stem == hr_i.stem
        self._len = len(self.blur_sharp_pairs)

        self.to_tensor = transforms.Compose(
            [transforms.ToTensor()]
        )
        self.preload = preload
        if self.preload:
            self.preload_img()
    
    def preload_img(self):
        self.blur_sharp_pairs = [
                (
                    self.to_tensor( self.cv_read(i)),
                    self.to_tensor( self.cv_read(j))
                ) for i,j in tqdm(self.blur_sharp_pairs)
            ]

    def cv_read(self, p:os.PathLike)->np.ndarray:
        return cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
    
    def __len__(self):
        return self._len

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        
        if self.preload:
            return self.blur_sharp_pairs[index]
        
        return self.to_tensor(self.cv_read(self.blur_sharp_pairs[index][0])), \
            self.to_tensor(self.cv_read(self.blur_sharp_pairs[index][1]))

class SR_View_Cmp_Dataset(SR_Dataset):
    
    def __init__(self, lr, hr, preload = False):
        super().__init__(lr, hr, preload)
    
    def preload_img(self):
        self.blur_sharp_pairs = [
            (
                self.to_tensor( self.cv_read(i)),
                str(j)
            ) for i,j in tqdm(self.blur_sharp_pairs)
        ]
    def __getitem__(self, index) -> tuple[torch.Tensor, str]:
        
        if self.preload:
            return self.blur_sharp_pairs[index]
        
        return self.to_tensor(self.cv_read(self.blur_sharp_pairs[index][0])), \
            str(self.blur_sharp_pairs[index][1])
