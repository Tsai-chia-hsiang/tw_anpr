import torch
from SR import SR_PRETRAINED_DIR, SR_FINETUNE_DIR
from SR.network import RRDBNet, UNetDiscriminatorSN
from SR.trainer import ESRGANModel
from SR.data import  SR_Dataset, SR_View_Cmp_Dataset
import gc
import cv2
import os
from pathlib import Path
from torch.utils.data import DataLoader
import random
import numpy as np
import torch
from tqdm import tqdm, trange
from torch.utils.tensorboard import SummaryWriter

def remove_old_tf_evenfile(directory:Path):
    if not directory.is_dir():
        return 
    
    old_board_file = list(
        filter(
            lambda x:'events.out.tfevents.' in x.name,
            [_ for _ in directory.iterdir() if _.is_file()]
        )
    )

    for i in old_board_file:
        print(f"remove {i}")
        os.remove(i)

def reproducible(seed:int = 891122):
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Ensure reproducibility with CUDA
    torch.cuda.manual_seed_all(seed)

    # Configure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@torch.no_grad()
def view_sr(model:RRDBNet, view_set:SR_View_Cmp_Dataset, dev:torch.device, save_dir:Path, batch_size:int=40):
    save_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    model.to(device=dev)
    val_loader = DataLoader(view_set, batch_size=batch_size)
    for lr, hr_path in tqdm(val_loader):
        sr = model.inference(lr.to(dev))
        for idx, si in enumerate(sr):
            img_name = Path(hr_path[idx]).name
            
            cv2.imwrite(
                save_dir/img_name,
                
                cv2.hconcat(
                    [
                        si, np.zeros((si.shape[0], 10, 3), dtype=np.uint8), 
                        cv2.imread(hr_path[idx])
                    ]
                )
            )


def forward_one_epoch(loader:DataLoader, model:ESRGANModel, pbar:tqdm, train=True, board:SummaryWriter=None, epoch:int=None) -> dict[str, float]:
    iters = 0
    eloss = None
    if train:
        torch.set_grad_enabled(True)
    else:
        torch.set_grad_enabled(False)

    N = len(loader.dataset)
    for lr, hr in loader:
        n_sample = len(lr)
        iters += n_sample
        losses = model.optimize_parameters(lq=lr, gt=hr, train=train)
        if pbar is not None:
            pbar.set_postfix(ordered_dict={'progress':f"{iters}/{N}","ploss":losses['l_g_percep']})
        
        if eloss is None:
            eloss = {k:0 for k in losses}
    
        for k in eloss:
            eloss[k] += losses[k]*n_sample
    

    prefix = "train" if train else "val" 
    for k in eloss:
        eloss[k] /= N
        if board is not None:
            board.add_scalar(f"{prefix}_{k}", eloss[k], epoch)
    
    return eloss

def main():
    
    reproducible()

    rrdbnet = RRDBNet(
        num_in_ch=3, num_out_ch=3, num_feat=64, 
        num_block=23, num_grow_ch=32, scale=2
    )
    rrdbnet.load_state_dict(
        torch.load(
            SR_PRETRAINED_DIR/"RealESRGAN_x2plus.pth",
            weights_only=True, map_location='cpu'
        )['params_ema']
    )
    udis = UNetDiscriminatorSN()

    udis.load_state_dict(
        torch.load(
           SR_PRETRAINED_DIR/"RealESRGAN_x2plus_netD.pth",
           weights_only=True, map_location='cpu'
        )['params']
    )
    data_root = Path("dataset/tw/sr")
    trainer = ESRGANModel(net_d=udis, net_g=rrdbnet, dev=torch.device("cuda:0"))
    
    train_set =  SR_Dataset(
        lr=data_root/"train"/"lr", 
        hr=data_root/"train"/"hr", 
        preload=True
    )
    train_loader = DataLoader(dataset=train_set, batch_size=25, shuffle=True)
    
    val_set = SR_Dataset(
        lr=data_root/"val"/"lr", 
        hr=data_root/"val"/"hr", 
        preload=True
    )
    val_loader = DataLoader(dataset=val_set, batch_size=25)

    view_set = SR_View_Cmp_Dataset(
        lr=data_root/"val"/"lr", 
        hr=data_root/"val"/"hr", 
        preload=True
    )

    remove_old_tf_evenfile(SR_FINETUNE_DIR)
    tensorboard_writer = SummaryWriter(log_dir=SR_FINETUNE_DIR)
    
    baseline = forward_one_epoch(
        loader=val_loader, model=trainer, 
        pbar=None, train=False
    )
    
    print(baseline)
    view_sr(
        model=rrdbnet, view_set=view_set,
        dev=torch.device("cuda:0"),
        save_dir=SR_FINETUNE_DIR/"baseline"
    )
    ploss = baseline['l_g_percep']
    pbar = trange(100)
    
    for e in pbar:

        train_eloss = forward_one_epoch(
            loader=train_loader, model=trainer, pbar=pbar, 
            train=True, board=tensorboard_writer, epoch=e
        )
   
        val_eloss = forward_one_epoch(
            loader=val_loader, model=trainer, pbar=pbar, 
            train=False, board=tensorboard_writer, epoch=e
        )
   
        if val_eloss['l_g_percep'] <= ploss:
            print(f"At {e} val percep loss from {ploss} -> {val_eloss['l_g_percep']}")
            torch.save(rrdbnet.state_dict(), SR_FINETUNE_DIR/f"g.pth")
            torch.save(udis.state_dict(), SR_FINETUNE_DIR/f"d.pth")
            ploss = val_eloss['l_g_percep']


    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    
    rrdbnet = RRDBNet(
        num_in_ch=3, num_out_ch=3, num_feat=64, 
        num_block=23, num_grow_ch=32, scale=2
    )
    rrdbnet = rrdbnet.load_state_dict(torch.load(SR_FINETUNE_DIR/f"g.pth", weights_only=True, map_location='cpu'))
    view_sr(
        model=rrdbnet, 
        view_set=view_set,
        dev=torch.device("cuda:0"),
        save_dir=SR_FINETUNE_DIR/f"val_best_ploss",
    )



if __name__ == "__main__":
    main()
