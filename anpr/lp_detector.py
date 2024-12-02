import os
from typing import Any
import math
from pathlib import Path
import torch
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results

_ANPR_DIR_ = Path(os.path.abspath(__file__)).parent

def crop_one(img:np.ndarray, bbox:list[int], cp:bool=True):
    y1,y2, x1,x2 = max(0, bbox[1]), min(img.shape[0], bbox[3]), \
                    max(0, bbox[0]), min(img.shape[1], bbox[2])
    c = img[y1:y2, x1:x2]  
    if not cp:
        return c
    return c.copy()

crop  = lambda img, bboxes, cp : [crop_one(img=img, bbox=b, cp=cp) for b in bboxes] 
flatten_2d_list = lambda lst: [i for j in lst for i in j]

def split_to_2d_list(lst: list, sp: list[int]) -> list[list]:
    acc = [0] + [sum(sp[:i]) for i in range(1, len(sp)+1)]
    return [lst[acc[i]:acc[i+1]] for i in range(len(sp))]

boxes_with_offset = lambda boxes, offset:[
    i + np.tile(offset_i, 2) if len(i) else i 
    for i, offset_i in zip(boxes, offset)
]

class LicensePlate_Detector():

    def __init__(self, model_path:os.PathLike=_ANPR_DIR_/f"anpr_v8.pt", gpu_id:int=0):
        self.model_path = model_path
        self.model = YOLO(self.model_path).to(device=torch.device(f"cuda:{gpu_id}"))

    @torch.no_grad()
    def __call__(self, imgs:np.ndarray|list[np.ndarray], conf:float=0.3, box_dtype:type=np.int32) -> list[np.ndarray]:
        
        """
        Args:
        
            - imgs: the image(s) that need to detect the license plate. Can be
                -  single image: a numpy ndarray in opencv manner
                -  a batch of images: list of numpy ndarray in opencv manner
                -  offset : where the coordinate start. If it's whole img, start from (0,0).
                            Otherwise you can give the crop top-left coordinate to normalize
        Return:
        
            list of the coordinate (xyxy) for each input image(s) 
            - first order : batch
            - 2nd order : [x,y,x,y] for that image 
        
        """
         
        results:Results = self.model(imgs, conf=conf, verbose=False, stream=True)
        plates_boxes = []
        for r in results:
            plates_boxes.append(
                np.asarray([b.xyxy.astype(box_dtype).squeeze() for b in r.cpu().numpy().boxes])
            )
        
        return plates_boxes
    
class Veh_Detector_with_LP_detection():
    
    def __init__(self, vehicle_model_path:str=None, lp_model_path:str=None, gpuid:int=0):

        self.lp_model_path = lp_model_path if lp_model_path is not None else Path(_ANPR_DIR_)/"anpr_v8.pt"
        self.vehicle_model_path = vehicle_model_path if vehicle_model_path is not None else Path(_ANPR_DIR_)/"yolov10l.pt"
        
        self.lp_model = LicensePlate_Detector(gpu_id=gpuid)
        self.vehicle_model = YOLO(self.vehicle_model_path).to(device=torch.device(f"cuda:{gpuid}"))

    @torch.no_grad()
    def __call__(self, imgs:np.ndarray|list[np.ndarray], lp_threshold:float=0.3) -> list[np.ndarray]:
        
        im0s = imgs if isinstance(imgs, list) else [imgs]
        vehs_results:list[Results] = self.vehicle_model(im0s, verbose=False, classes=[2, 3, 5, 7], stream=True)
        veh_boxes = [r.boxes.cpu().numpy().xyxy.astype(np.int32) for r in vehs_results]
        sp = [len(_) for _ in veh_boxes]
        veh_crops = flatten_2d_list([crop(img=im, bboxes=b, cp=False) for im, b in zip(im0s, veh_boxes)])
        
        if len(veh_crops) == 0:
            return [[[]] for _ in range(len(im0s))]
        
        lp_boxes = self.lp_model(imgs=veh_crops, conf=lp_threshold)
        offset = np.vstack(veh_boxes)
        lp_boxes = boxes_with_offset(boxes=lp_boxes, offset=offset[:, :2])
        lp_boxes = split_to_2d_list(lp_boxes, sp=sp)
        
        return lp_boxes
        