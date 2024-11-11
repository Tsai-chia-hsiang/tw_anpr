from pathlib import Path
import torch
from logging import Logger
from typing import Optional

def load_networks(net:torch.nn.Module, pretrained_ckpt:Path, logger:Optional[Logger]=None) -> None:
    
    if pretrained_ckpt is None:
        if logger is not None:
            logger.info(f"pretrained_ckpt is None, will not load")
        else:
            print(f"pretrained_ckpt is None, will not load")
        return 
    
    if not pretrained_ckpt.is_file():
        if logger is not None:
            logger.info(f"No such {pretrained_ckpt} file, will not load")
        else:
            print(f"No such {pretrained_ckpt} file, will not load")
        return 
    
    if logger is not None:
        logger.info(f"loading the model from {pretrained_ckpt}")
    else:
        print(f"loading the model from {pretrained_ckpt}")
    
    state_dict = torch.load(pretrained_ckpt, map_location='cpu', weights_only=True)
    
    if hasattr(state_dict, '_metadata'):
        del state_dict._metadata
    
    net.load_state_dict(state_dict)