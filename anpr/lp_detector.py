import os
from pathlib import Path
import torch
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results

_ANPR_DIR_ = Path(os.path.abspath(__file__)).parent



class LicensePlate_Detection_YOLOV8:

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