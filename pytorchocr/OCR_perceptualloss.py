import torch
from typing import Any, Literal
import math
import torch.nn.functional as F
from pathlib import Path
from .base_ocr_v20 import BaseOCRV20
from torch.nn import L1Loss, CrossEntropyLoss, MSELoss, KLDivLoss
from . import TORCH_OCR_DIR
import yaml


def decide_target_img_size(input_hw:tuple[int, int],sys_hw:tuple[int,int]=(48, 320), limited_max_width=1280, limited_min_width=16) -> tuple[int, int]:
    
    w = input_hw[1]
    h = input_hw[0]

    src_wh_ratio = w/h
    imgW = int(src_wh_ratio*sys_hw[0])
    imgW = max(min(imgW, limited_max_width), limited_min_width)

    ratio = w/float(h)
    ratio_imgH = max(math.ceil(sys_hw[0]*ratio), limited_min_width)
    if ratio_imgH > imgW:
        resized_w = imgW
    else:
        resized_w = int(ratio_imgH)
    
    return sys_hw[0], resized_w


def load_yml(fp):
    with open(fp, 'r', encoding='utf-8') as f:
        content = yaml.safe_load(f)
    return content

class TextRec_Loss(torch.nn.Module):

    def __init__(
            self, input_hw:tuple[int, int], 
            config_file:Path=TORCH_OCR_DIR/"config"/"en_PP-OCRv4_rec_arch.yml", 
            pretrain:Path=TORCH_OCR_DIR/"pretrained"/"en_ptocr_v4_rec_infer.pth", 
            sys_hw:tuple[int, int]=(48, 320), 
            limited_max_width=1280, limited_min_width=16,
            loss_type:Literal['L1', "MSE", "CE", "KL"]="L1",
            **kwargs
        ) -> None:
        
        super(TextRec_Loss, self).__init__(**kwargs) 
        
        self.model = BaseOCRV20(config=load_yml(config_file), **(kwargs|{'out_channels':97, 'char_num':97}))

        self.model.load_state_dict(torch.load(pretrain, map_location='cpu', weights_only=True))

        self.processing_hw = decide_target_img_size(
            input_hw=input_hw, sys_hw=sys_hw,
            limited_max_width=limited_max_width, 
            limited_min_width=limited_min_width
        )
        self.zero_padding = torch.nn.ZeroPad2d((0, sys_hw[1]-self.processing_hw[1], 0, 0))
        self.loss = None
        
        match loss_type:
            case 'L1':
                self.loss = L1Loss()
            case 'MSE':
                self.loss = MSELoss()
            case 'CE':
                raise NotImplementedError()
            case "KL":
                raise NotImplementedError()
            case _:
                raise KeyError(f"Not support {loss_type} loss")
        
        for p in self.model.net.parameters():
            p.requires_grad = False
        
    def to(self, device:torch.device, **kwargs):
        self.model.net.to(device=device, **kwargs)
        super().to(device=device, **kwargs)
    
    def forward(self, x:torch.Tensor, gt:torch.Tensor, logit_w:float=1.0, backbone_w:float=-1, neck_w:float=-1) -> torch.Tensor:
        """
        feature: dict of tensors
            - back_out: backbone output
            - neck_out: neck output 
                - NOTE: ppv4 doesn't have neck, so it's the same as ctc_neck from head_out  
            - head_out: head output
                - ctc: logit
                - res: same as ctc
                - ctc_neck: the features before go into linear classifer layer
        """
        pgt = self.preprocessing(gt)
        px = self.preprocessing(x)
        
        gt_feature = self.model.net(pgt.detach())
        px_feature = self.model.net(px)

        l:torch.Tensor = 0
        if logit_w > 0:
            l += logit_w*self.loss(input=px_feature['head_out']['ctc'], target=gt_feature['head_out']['ctc'])
        if backbone_w > 0:
            l += backbone_w*self.loss(input=px_feature['backbone_out'], target=gt_feature['backbone_out'])
        if neck_w > 0:
            l += neck_w*self.loss(input=px_feature['head_out']['ctc_neck'], target=gt_feature['head_out']['ctc_neck'])

        return l 


    def preprocessing(self, x:torch.Tensor) -> torch.Tensor:
        """
        x: image tensor with shape : (B,C,H,W) 
        """
        output_tensor = F.interpolate(x[:, [2, 1, 0], :, :], size=self.processing_hw, mode='bilinear', align_corners=False)
        output_tensor = (output_tensor-0.5)/0.5
        return self.zero_padding(output_tensor)


