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
from anpr import  OCR_Evaluator
from logger import remove_old_tf_evenfile, print_infomation, get_logger
from torchtool import reproducible
from argparse import ArgumentParser

evar = OCR_Evaluator()

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


def forward_one_epoch(loader:DataLoader, model:ESRGANModel, pbar:tqdm, train=True, board:SummaryWriter=None, epoch:int=None, logger=None) -> dict[str, float]:
    iters = 0
    eloss = None
    if train:
        torch.set_grad_enabled(True)
    else:
        torch.set_grad_enabled(False)

    N = len(loader.dataset)
    gt = []
    pred = []
    for lr, hr, lr_path, sr_path, lp_num in loader:
        n_sample = len(lr)
        iters += n_sample
        losses = model.optimize_parameters(lq=lr, gt=hr, train=train)
        
        with torch.no_grad():
            model.net_g.eval()
            sr_imgs = model.net_g.inference(x=lr.to(model.dev))
            gt += lp_num
            pred += [SR_Dataset.lp_ocr(sr_img)[1] for sr_img in sr_imgs]

        
        if pbar is not None:
            pbar.set_postfix(ordered_dict={'progress':f"{iters}/{N}", 'lr':model.current_lr})
        
        if eloss is None:
            eloss = {k:0 for k in losses}
    
        for k in eloss:
            eloss[k] += losses[k]*n_sample
    

    prefix = "train" if train else "val" 
    acc = evar(pred=pred, gth=gt, metrics='lcs')
    eloss =  eloss | {f'{prefix}_ocr_acc': acc}
    
    for k in eloss:
        if 'ocr_acc' not in k:
            eloss[k] /= N
        if board is not None:
            board.add_scalar(k, eloss[k], epoch)
    print_infomation(eloss, logger=logger)

    return eloss

def main(args):
    
    reproducible(seed=args.seed)
    project_dir:Path = SR_FINETUNE_DIR/args.save_project
    project_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger(name=__file__, file=project_dir/f"training.log")

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

    trainer = ESRGANModel(
        net_d=udis, net_g=rrdbnet,
        dev=torch.device(args.devices),
        lr=args.lr, 
        MultiStepLR_mileston=args.epoch_stage[:-1],
        gamma=args.gamma,
        g_loss_w=args.g_loss_w,
        p_loss_w=args.p_loss_w,
        p_losses=args.p_losses,
        ocr_p_w={
            'logit_w':args.ocr_p_logit_w, 
            'backbone_w':args.ocr_p_backbone_w, 
            'neck_w':args.ocr_p_neck_w
        }
    )
    
    train_set =  SR_Dataset(
        lr=args.train_root/"lr", 
        hr=args.train_root/"hr", 
        preload=True,
        label_file=args.train_root/f"labels.json"
    )
    train_loader = DataLoader(dataset=train_set, batch_size=args.batch, shuffle=True)
    
    val_set = SR_Dataset(
        lr=args.val_root/"lr", 
        hr=args.val_root/"hr", 
        label_file=args.val_root/"labels.json",
        preload=True
    )
    val_loader = DataLoader(dataset=val_set, batch_size=args.batch)

    view_set = SR_View_Cmp_Dataset(
        lr=args.val_root/"lr", 
        hr=args.val_root/"hr", 
        preload=True
    )

    remove_old_tf_evenfile(project_dir)
    tensorboard_writer = SummaryWriter(log_dir=project_dir)
    
    baseline = forward_one_epoch(
        loader=val_loader, model=trainer, 
        pbar=None, train=False, logger=logger,
        epoch=0, board=tensorboard_writer
    )
    
    ocr_acc = baseline['val_ocr_acc']
    pbar = trange(sum(args.epoch_stage))
    
    for e in pbar:

        train_eloss = forward_one_epoch(
            loader=train_loader, model=trainer, pbar=pbar, 
            train=True, board=tensorboard_writer, epoch=e,
            logger=logger
        )
   
        val_eloss = forward_one_epoch(
            loader=val_loader, model=trainer, pbar=pbar, 
            train=False, board=tensorboard_writer, epoch=e+1,
            logger=logger
        )
        trainer.update_lr()
        print_infomation(val_eloss['val_ocr_acc'], logger=logger)
        if val_eloss['val_ocr_acc'] >= ocr_acc:
            print_infomation(f"At {e} val ocr from {ocr_acc} -> {val_eloss['val_ocr_acc']}", logger=logger)
            torch.save(rrdbnet.state_dict(), project_dir/f"g.pth")
            torch.save(udis.state_dict(), project_dir/f"d.pth")
            ocr_acc = val_eloss['val_ocr_acc']

    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    if (project_dir/f"g.pth").is_file():
        print_infomation(f"Have trained a model that exceed the baseline", logger=logger)
        rrdbnet = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64, 
            num_block=23, num_grow_ch=32, scale=2
        )
        rrdbnet.load_state_dict(torch.load(project_dir/f"g.pth", weights_only=True, map_location='cpu'))
        view_sr(
            model=rrdbnet, 
            view_set=view_set,
            dev=torch.device(args.devices),
            save_dir=project_dir/f"val_best_ocr",
        )
    else:
        print_infomation("Can't exceed baseline, give up", logger=logger)



if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--save_project", type=str, default="ocr")
    parser.add_argument("--devices", type=str, default='0')
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--g_loss_w", type=float, default=0.1)
    parser.add_argument("--p_losses",type=str, nargs='+', default=['ocr', 'vgg'])
    
    parser.add_argument("--p_loss_w", type=float, default=1)

    parser.add_argument("--ocr_p_logit_w", type=float, default=1.0)
    parser.add_argument("--ocr_p_backbone_w", type=float, default=1.0)
    parser.add_argument("--ocr_p_neck_w", type=float, default=1.0)

    
    parser.add_argument("--epoch_stage",nargs='+', type=int, default=[100, 50])
    parser.add_argument("--batch", type=int, default=25)
    parser.add_argument("--train_root", type=Path, default=Path("dataset")/"tw"/"sr"/"train")
    parser.add_argument("--val_root", type=Path, default=Path("dataset")/"tw"/"sr"/"val")
    parser.add_argument("--seed", type=int, default=891122)

    args = parser.parse_args()
    args.epoch_stage = list(args.epoch_stage)
    if args.devices.isdigit():
        args.devices = f"cuda:{args.devices}" 
    
    print(args)
    main(args)
    

