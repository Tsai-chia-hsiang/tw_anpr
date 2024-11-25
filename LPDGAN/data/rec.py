from pathlib import Path
import numpy as np
import torch
import cv2
from paddleocr import PaddleOCR
import logging
from paddleocr.ppocr.utils.logging import get_logger
import easyocr
from torchvision.transforms import transforms, ToTensor, Normalize
from .aug import normalize_brightness

_paddle_logger = get_logger()
_paddle_logger.setLevel(logging.ERROR)

_paddleocr_model_:PaddleOCR = None
_easyocr_crnn_:torch.nn.Module = None
_easyocr_transform = None

__all__ = ["EXT_MAP"]

def paddle_text_info(img:Path, on_size:tuple[int, int]=(224, 112)) -> torch.Tensor:
    global _paddleocr_model_         
    
   
    if _paddleocr_model_ is None:
        _paddleocr_model_ = PaddleOCR(use_angle_cls=True, lang="en")
 
    i = normalize_brightness(cv2.resize(cv2.imread(img), on_size))
    logit = _paddleocr_model_.ocr(i, cls=True, det=False)[0][0][3]
    return torch.from_numpy(logit)
    

def get_easy_ocr_rcnn():
    reader = easyocr.Reader(['en'], gpu=False)  # You can set gpu=True if you have a GPU
    # Access the recognition model (CRNN)
    fe = reader.recognizer
    for param in fe.parameters():
        param.requires_grad = False
    fe.eval()
    return fe  


def easy_ocr_crnn_text_info(img:Path, on_size:tuple[int, int]=(224, 112)) -> torch.Tensor:
    global _easyocr_crnn_, _easyocr_transform
    if _easyocr_crnn_ is None:
        _easyocr_crnn_ = get_easy_ocr_rcnn()
    if _easyocr_transform is None:
        _easyocr_transform = transforms.Compose(
            [
                ToTensor(),
                Normalize(mean=[0.5], std=[0.5])
            ]
        )
    i = cv2.cvtColor(cv2.resize(cv2.imread(img), on_size),cv2.COLOR_BGR2GRAY)
    logit = _easyocr_crnn_(_easyocr_transform(i).unsqueeze(0),text=None)[0]
    prob = torch.nn.functional.softmax(logit, dim=-1)
    return prob

EXT_MAP = {
    'easyocr':easy_ocr_crnn_text_info,
    'paddleocr':paddle_text_info
}