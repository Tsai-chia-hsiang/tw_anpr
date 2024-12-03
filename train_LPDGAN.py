"""
Refactoring from https://github.com/haoyGONG/LPDGAN.git
"""
import gc
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import shutil
import random
import time
from tqdm import tqdm, trange
from argparse import ArgumentParser, Namespace
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import json
from torch.utils.data import DataLoader
from pathlib import Path
from LPDGAN import LP_Deblur_Dataset, LP_Deblur_OCR_Valiation_Dataset, LPDGAN_Trainer, LPDGAN_DEFALUT_CKPT_DIR, LPD_OCR_ACC_Evaluator
from logger import get_logger, print_infomation
from imgproc_utils import L_CLAHE
from torchtool import reproducible 


def main(args:Namespace):
    
    # if there is a same name, replace it by delete the old one first
    if args.model_save_root.is_dir():
        shutil.rmtree(args.model_save_root)
    
    args.model_save_root.mkdir(parents=True, exist_ok=True)
 
    logger = get_logger(name=__name__, file=args.model_save_root/"training.log")
    tensorboard_writer = SummaryWriter(log_dir=args.model_save_root)
    
    print_infomation(f"{args}", logger=logger)
    
    reproducible(seed=args.seed)
    
    dataset_root:Path = args.data_root
    dataset = LP_Deblur_Dataset(data_root = dataset_root, blur_aug = args.blur_aug, preload=args.preload)

    logger.info(f'The number of training pairs = {len(dataset)}')
    trainloader = DataLoader(
        dataset=dataset, batch_size=args.batch_size, 
        shuffle=True, num_workers=int(args.num_threads),
    )
    warm_up_loader = DataLoader(
        dataset=dataset, batch_size=args.warm_up_batch_size, 
        shuffle=True, num_workers=int(args.num_threads)
    )
    val_dataset = LP_Deblur_OCR_Valiation_Dataset.build_dataset(
        dataroot=args.val_data_root,
        label_file=args.label_file,
    )
    validator:LPD_OCR_ACC_Evaluator = None
    val_loader:DataLoader = None
    baseline = {
        'cer':args.ocr_cer_baseline,
        'lcs':args.ocr_lcs_baseline
    }
    if val_dataset is not None:
        logger.info(f"validation set : {args.val_data_root}, label:{args.label_file}, num: {len(val_dataset)}")
        val_loader = DataLoader(dataset=val_dataset, batch_size=args.val_batch, shuffle=False)
        validator = LPD_OCR_ACC_Evaluator(ocr_preprocess=None, current_best=0.0)

    lpdgan = LPDGAN_Trainer(
        logger=logger, epochs_policy= {
            'n_epochs':args.n_epochs + args.D_warm_up,
            'n_epochs_decay':args.n_epochs_decay,
            'lr_decay_iters':args.lr_decay_iters
        },
        pretrained_weights=args.pretrained_weights,
        gan_mode=args.gan_mode,
        lr=args.lr, lr_policy=args.lr_policy, lambda_L1=args.lambda_L1, 
        gpu_id=args.gpu_id,
        D_warm_up=args.D_warm_up,
        load_G=args.load_G, load_D=args.load_D
    )
    
    epoch_loss = None
    pbar = trange(1, args.D_warm_up + args.n_epochs + args.n_epochs_decay + 1)
    for epoch in pbar:
        epoch_val_flag = (epoch > args.D_warm_up) and (epoch - args.D_warm_up) % args.save_epoch_freq == 0
        epoch_iters = 0
         
        target_loader = trainloader if epoch > args.D_warm_up else warm_up_loader
        
        old_lr, lr = lpdgan.update_learning_rate()
        
        for data in target_loader:
            n = len(data['A_paths'])
            epoch_iters += n
            
            lpdgan.optimize_parameters(input_x=data, step=epoch-1)
            pbar.set_postfix(ordered_dict={'iters':f"{epoch_iters}/{len(dataset)}", "stage":lpdgan.stage, "lr":f"{lr}<-{old_lr}"})
            
            if epoch_val_flag:
                iter_loss = lpdgan.get_current_losses()
                if epoch_loss is None:
                    epoch_loss = iter_loss.copy()
                else:
                    for k in iter_loss:
                        epoch_loss[k] += iter_loss[k]*n

        if epoch_val_flag:
            
            for k in epoch_loss:
                epoch_loss[k] /= len(dataset)

            tensorboard_writer.add_scalars("Loss",epoch_loss, epoch)
            epoch_loss=None
            gc.collect()
            
            if args.not_save:
                continue
            
            if validator is not None:
                validator(
                    lpdgan=lpdgan, val_loader=val_loader,
                    save_dir = args.model_save_root ,
                    iters=epoch, board=tensorboard_writer,
                    logger=logger, baseline=baseline
                )
            else:
                epoch_save = args.model_save_root /f'epoch_{epoch}'
                lpdgan.save_networks(save_dir=epoch_save)
          
     
    lpdgan.save_networks(save_dir = args.model_save_root/'latest')
    print_infomation(f"Last step optimizers and schedulers are saved at { args.model_save_root/'latest'}", logger=logger)

    if validator is not None:
        with open(args.model_save_root/f"val_metrics.json", "w+") as jf:
            json.dump(validator.hist, jf, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    
    parser = ArgumentParser()

    # Data path
    parser.add_argument("--data_root", type=Path, default=Path("./dataset")/"tw"/"new")
    parser.add_argument("--blur_aug", nargs='+', type=str, default='all')
    parser.add_argument("--txt_cached", type=Path, default=None)
    parser.add_argument("--txt_extract", type=str, default='paddleocr')
    parser.add_argument("--preload", action='store_true')
    parser.add_argument("--val_data_root", type=Path, default=None)
    parser.add_argument("--label_file", type=Path, default=None)
    parser.add_argument("--val_batch", type=int, default=40)
    
    parser.add_argument("--ocr_lcs_baseline", type=float, default=0.51)
    parser.add_argument("--ocr_cer_baseline", type=float, default=0.49)

    parser.add_argument('--num_threads', default=0, type=int, help='# threads for loading data')
    parser.add_argument("--gpu_id", type=str, default='0')
    parser.add_argument("--seed", type=int, default=891122)

    # Batch size & epochs & lr scheduler
    parser.add_argument("--batch_size", type=int, default=40)
    parser.add_argument('--D_warm_up', type=int, default=2)
    parser.add_argument("--n_epochs", type=int, default=100, help='number of epochs with the initial learning rate')
    parser.add_argument("--n_epochs_decay", type=int, default=200, help='number of epochs to linearly decay learning rate to zero')
    parser.add_argument("--lr_decay_iters", type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
    parser.add_argument('--lr_policy', type=str, default='linear',help='learning rate policy. [linear | step | plateau | cosine]')
    parser.add_argument('--warm_up_batch_size', type=int, default=50)
    # save mode 
    parser.add_argument('--not_save', action='store_true')
    parser.add_argument('--save_epoch_freq', type=int, default=5)
    parser.add_argument("--model_save_root", type=Path, default=LPDGAN_DEFALUT_CKPT_DIR)
   
    # load pretrained model
    parser.add_argument('--pretrained_weights', type=Path, default=None)
    parser.add_argument('--checkpoint_dir', type=Path, default=None, help='Using the optimizers and schedulers saved from previous work to keep training')
    parser.add_argument('--load_G', action='store_true')
    parser.add_argument('--load_D', action='store_true')
    
    # lr
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
    
    # GAN
    parser.add_argument('--gan_mode', type=str, default='wgangp')
    
    # Loss function
    parser.add_argument("--ocr_perceptual", action='store_false')
    parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
    parser.add_argument('--txt_loss', type=str, default='l1')

    args = parser.parse_args()
    if args.blur_aug == "all":
        args.blur_aug = [_.name for _ in args.data_root.iterdir() if _.name != "sharp" and _.is_dir()]
    
    # print(args)
    # _ = input("ok ?")
    main(args)
