import os
import cv2
from pathlib import Path
from typing import Optional, Literal
from torch.utils.data import Dataset
import torch
from torchvision import transforms
import json
import easyocr
from logging import Logger
import torch.nn as nn
from .sp import Spatial_Pyramid_cv2

def get_easy_ocr_rcnn():
    reader = easyocr.Reader(['en'], gpu=False)  # You can set gpu=True if you have a GPU
    # Access the recognition model (CRNN)
    fe = reader.recognizer
    for param in fe.parameters():
        param.requires_grad = False
    fe.eval()
    return fe  


flatten2D = lambda  nested_list: [item for sublist in nested_list for item in sublist]

class LP_Deblur_OCR_Valiation_set(Dataset):
    def __init__(self, imgs:list[Path], labels:list[Path], org_size:tuple[int,int]=(224,112), on_brightness:Optional[int]=180):
        super().__init__()
        self.imgs = imgs
        self.labels = labels
        self.sp = Spatial_Pyramid_cv2(org_size=org_size, origin_brightness=on_brightness)
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, index) -> dict[str, torch.Tensor|str]:
        blur_img = cv2.imread(self.imgs[index]) 
        r = self.sp(img=blur_img, L=3, map_key="A")
        r['gth'] = self.labels[index]
        return r
    
    @classmethod
    def make_val_ocr_dataset(cls, val_dataset_root:Optional[Path]=None, label_file:Optional[Path]=None, org_size:tuple[int,int]=(224,112), on_brightness:Optional[int]=180, logger:Optional[Logger]=None) -> "LP_Deblur_OCR_Valiation_set"|None:
        
        if val_dataset_root is None or label_file is None:
            if logger is not None:
                logger.info("No valiation dataset, return None as validation set")
            else:
                print("No valiation dataset, return None as validation set")
            return None
        
        if not val_dataset_root.is_dir():
            if logger is not None:
                logger.info(f"No such {val_dataset_root} a dir, return None as validation set")
            else:
                print(f"No such {val_dataset_root} a dir, return None as validation set")
            return None
        
        if not label_file.is_file():
            if logger is not None:
                logger.info(f"No such {label_file} a label file, return None as validation set")
            else:
                print(f"No such {label_file} a label file, return None as validation set")
            return None
        
        labels:dict[str, str] = None
        with open(label_file, "r") as f:
            labels = json.load(f)
        
        imgs = [val_dataset_root/i for i in list(labels.keys())]
        plate_num = list(labels.values())

        return cls(imgs=imgs, labels = plate_num, org_size=org_size, on_brightness=on_brightness)




class LPBlurDataset(Dataset):
    
    def __init__(self, data_root:Path, blur_aug:list[str], mode:Literal['train', 'test']="train", org_size:tuple[int, int]= (224, 112), on_brightness:Optional[int]=None) -> None:
        
        super().__init__()
        self.mode = mode
        self.org_size = org_size
        self.text_crnn, self.advp = None ,None
        self.need_gth = self.mode == "train"
        if self.need_gth:
            self.text_crnn = get_easy_ocr_rcnn()
            self.advp = nn.AdaptiveMaxPool2d((21, 1))
        self.blur_aug = blur_aug
        self.sharp_root = Path(data_root)/"sharp"
        imgids = [_.name for _ in (self.sharp_root).glob("*.jpg")]

        #_ = input(f"using {self.blur_aug} as blur pairs ? ")
        
        self.sharp_blur_pairs = [
            [
                (self.sharp_root/imgid if self.need_gth else None, 
                Path(data_root)/f"{b}"/imgid) 
                for imgid in imgids
            ] 
            for b in self.blur_aug
        ]
        
        self.sharp_blur_pairs = flatten2D(self.sharp_blur_pairs)
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

        blur_path = ps[1] if self.need_gth else ps
        blur_img = cv2.imread(blur_path)  
        r = self.sp(img=blur_img, L=3, map_key="A")
        r['A_paths'] = str(blur_path)
        
        if self.need_gth:
            sharp_image = cv2.imread(ps[0])
            r['plate_info'] = self.get_text_info(sharp=sharp_image)
            r = r | self.sp(img=sharp_image, L=4, map_key="B")
            r['B_paths']= str(ps[1])
        
        return r
        
    @torch.no_grad()
    def get_text_info(self, sharp)->torch.Tensor:
        
        sharp_im = transforms.ToTensor()(cv2.cvtColor(sharp, cv2.COLOR_BGR2GRAY)).unsqueeze(0) 
        visual_feature = self.text_crnn.FeatureExtraction(sharp_im)
        visual_feature = self.text_crnn.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))
        visual_feature = visual_feature.squeeze(3)

        """ Sequence modeling stage """
        contextual_feature = self.text_crnn.SequenceModeling(visual_feature).unsqueeze(1)
        text_f = self.advp(contextual_feature).squeeze()  # Shape: (batch_size, 1, 21, 1)
        return text_f