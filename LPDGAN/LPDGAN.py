from typing import Optional, Literal
from pathlib import Path
import os
import torch.nn as nn
import numpy as np
import torch
from .models import SwinTransformer_Backbone, get_config_or, load_networks
from .data import tensor2img, Spatial_Pyramid_cv2

_LPDGAN_DIR_ = Path(os.path.abspath(__file__)).parent
LPDGAN_DEFALUT_CKPT_DIR = _LPDGAN_DIR_/"checkpoints"

check_gpu_id = lambda gpu_id: gpu_id > -1 and torch.cuda.is_available() and gpu_id < torch.cuda.device_count()

class SwinTrans_G(nn.Module):
    
    def __init__(self, pretrained_ckpt:Optional[Path]=None, gpu_id:int=0, mode:Literal["train", "inference"]="train", on_size:tuple[int, int]=(224, 112)):
        
        super(SwinTrans_G, self).__init__()
        self.mode = mode
        self.device = torch.device(f'cuda:{gpu_id}') if check_gpu_id(gpu_id=gpu_id) else torch.device('cpu')
        config_su = get_config_or()
        self.netG = SwinTransformer_Backbone(config_su)
        self.model_name = 'G'
        load_networks(pretrained_ckpt=pretrained_ckpt, net=self.netG)
        self.netG.to(device=self.device)
        self.on_size = on_size
        self.inference_aug = Spatial_Pyramid_cv2(org_size=self.on_size) 
        
        match self.mode:
            case "train":
                self.train()
            case "inference":
                self.eval()

    def forward(self, x:dict[str, torch.Tensor]) -> torch.Tensor:

        fake_B, fake_B1, fake_B2, fake_B3, plate1, plate2 = self.netG(
            x['A0'].to(self.device), x['A1'].to(self.device), x['A2'].to(self.device)
        )
        return fake_B, fake_B1, fake_B2, fake_B3, plate1, plate2
    
    @torch.no_grad()
    def inference(self, x:np.ndarray, to_cv2:bool=True) -> np.ndarray:

        fake_B, _ ,_ ,_ ,_, _ = self(self.inference_aug(img=x, map_key='A', L=3, to_batch=True))
        
        return tensor2img(input_image=fake_B, to_cv2=to_cv2)
