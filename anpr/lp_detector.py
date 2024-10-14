import os
from typing import Any
from pathlib import Path
import torch
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results

_ANPR_DIR_ = Path(os.path.abspath(__file__)).parent



class LicensePlate_Detector():

    def __init__(self, model_path:os.PathLike=_ANPR_DIR_/f"anpr_v8.pt", gpu_id:int=0):
        self.model_path = model_path
        self.model = YOLO(self.model_path).to(device=torch.device(f"cuda:{gpu_id}"))

    def __call__(self, imgs:np.ndarray|list[np.ndarray], conf:float=0.3) -> list[list[int]]:
        """
        Args:
        --
        - imgs: the image(s) that need to detect the license plate. Can be
            -  single image: a numpy ndarray in opencv manner
            -  a batch of images: list of numpy ndarray in opencv manner
        
        Return:
        --
        list of the coordinate (xyxy) for each input image(s) 
        - first order : batch
        - 2nd order : [x,y,x,y] for that image 
        """
        
        results:Results = self.model(imgs, conf=conf, verbose=False)
        plates = [
            [b.xyxy[0].astype(np.int32).tolist() 
             for b in r.cpu().numpy().boxes] 
            for r in results
        ]
        #print(plates)
        return plates
    


class Veh_Detector_with_LP_detection():
    
    def __init__(self, vehicle_model_path:str=None, lp_model_path:str=None, gpuid:int=0):

        self.lp_model_path = lp_model_path if lp_model_path is not None else Path(_ANPR_DIR_)/"anpr_v8.pt"
        self.vehicle_model_path = vehicle_model_path if vehicle_model_path is not None else Path(FILE_DIR)/"yolov10l.pt"
        
        self.lp_model = YOLO(self.lp_model_path).to(device=torch.device(f"cuda:{gpuid}"))
        self.vehicle_model = YOLO(self.vehicle_model_path).to(device=torch.device(f"cuda:{gpuid}"))

    def __call__(self, img:np.ndarray, lp_threshold:float=0.3, vehicle_crop:bool=False, lp_crop:bool=True) -> list[dict[str, Any]]:
        
        ret = []
        vehs = self.vehicle_model(img, verbose=False, classes=[2, 3, 5, 7])[0].boxes
        
        for v in vehs:
            ri = {}

            x0, y0, xe, ye = v.xyxy.cpu().numpy()[0]
            x0, y0, xe, ye =  int(x0), int(y0), int(xe), int(ye)

            a_veh = img[y0:ye, x0:xe]

            ri['vehicle_coors'] = [x0, y0, xe, ye]
            if vehicle_crop:
                ri['vehicle_crop'] = a_veh
            
            lp_result = self.lp_model(a_veh, verbose=False)[0]
            if len(lp_result):
                
                lp = lp_result.boxes[0]
                x1, y1, x2, y2 = lp.xyxy.cpu().numpy()[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                conf = lp.conf
                
                if conf >= lp_threshold:
                    if lp_crop:
                        ri['lp_crop'] = a_veh[y1:y2, x1:x2]
                    
                    ri['lp_coors'] = [x1+x0, y1+y0, x2+x0, y2+y0]
            
            ret.append(ri)

        return ret