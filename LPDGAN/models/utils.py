from pathlib import Path
import torch
from logging import Logger
from typing import Optional
"""import sys
import os
sys.path.append(str(Path(os.path.abspath(__file__)).parent/"logger"))"""
from ..logger import print_infomation

def load_networks(net:torch.nn.Module, pretrained_ckpt:Path, logger:Optional[Logger]=None, show_log=True) -> None:
    
    if show_log:
        if pretrained_ckpt is None:
            print_infomation(f"pretrained_ckpt is None, will not load", logger=logger)
            return 
        
    if not pretrained_ckpt.is_file():
        if show_log:
            print_infomation(f"No such {pretrained_ckpt} file, will not load", logger=logger)
            return 
    
    if show_log:
       print_infomation(f"load weight from {pretrained_ckpt}", logger=logger)
    
    state_dict = torch.load(pretrained_ckpt, map_location='cpu', weights_only=True)
    
    if hasattr(state_dict, '_metadata'):
        del state_dict._metadata
    
    net.load_state_dict(state_dict)