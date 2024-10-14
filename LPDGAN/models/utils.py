from pathlib import Path
import torch

def load_networks(net:torch.nn.Module, pretrained_ckpt:Path) -> None:
    
    if pretrained_ckpt is None:
        print(f"{pretrained_ckpt} is None, will not load")
        return 
    
    if not pretrained_ckpt.is_file():
        print(f"No such {pretrained_ckpt} file, will not load")
        return 
    
    print(f"loading the model from {pretrained_ckpt}")
    state_dict = torch.load(pretrained_ckpt, map_location='cpu', weights_only=True)
    
    if hasattr(state_dict, '_metadata'):
        del state_dict._metadata
    
    net.load_state_dict(state_dict)