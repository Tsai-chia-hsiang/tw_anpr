"""
Refactoring from https://github.com/haoyGONG/LPDGAN.git
"""
import gc
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import shutil
import random
from typing import Optional
from logging import Logger
import time
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import cv2
import json
from torch.utils.data import DataLoader
from pathlib import Path
from LPDGAN import LP_Deblur_Dataset, LP_Deblur_OCR_Valiation_Dataset, LPDGAN_Trainer, LPDGAN_DEFALUT_CKPT_DIR, LPD_OCR_ACC_Evaluator
from LPDGAN.logger import get_logger
from imgproc_utils import L_CLAHE

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

def evaluation_and_save(
    lpdgan:LPDGAN_Trainer, val_loader:DataLoader, eva:LPD_OCR_ACC_Evaluator, 
    save_dir:Path, logger:Logger,iters:int, baseline:dict[str, float], 
    epoch:Optional[int]=None, board:Optional[SummaryWriter]=None,
    keep_training:bool=True
) -> float:
    if val_loader is not None:
        # Having a validaiton dataset.
        # Will first validate the model and decide wether to save 
        val_msg = f"validation at iteration {iters}"
        if epoch is not None:
            val_msg += f"(End of epoch {epoch:3d})"
        
        logger.info(val_msg)
        
        lpdgan.netG.eval()
        last_best = eva.current_best
        acc = eva.val_LP_db_dataset(val_loader=val_loader, swintrans_g=lpdgan.netG)
        update_status = eva.update(accs=acc)
        for k in acc:
            logger.info(f"{k}: baseline: {baseline[k]} | acc : {acc[k]} | last best: {last_best[k]} | {update_status[k]}")
            if board is not None:
                board.add_scalars(
                    k, {"train":acc[k], "baseline":baseline[k]},
                    iters
                )
            if update_status[k] == LPD_OCR_ACC_Evaluator.update_signal:
                lpdgan.save_networks(save_dir=save_dir/f"{k}")
        if keep_training:
            lpdgan.netG.train()
        
        return acc
    else:
        # no validation, directly save
        logger.info(f"no validation set, directly save to {save_dir}")
        lpdgan.save_networks(save_dir=save_dir)
        return -1.0

def main(args:Namespace):
    
    # if there is a same name, replace it by delete the old one first
    if args.model_save_root.is_dir():
        shutil.rmtree(args.model_save_root)
    
    args.model_save_root.mkdir(parents=True, exist_ok=True)
 
    logger = get_logger(name=__name__, file=args.model_save_root/"training.log")
    tensorboard_writer = SummaryWriter(log_dir=args.model_save_root)
    
    logger.info(f"{args}")
    
    reproducible(seed=args.seed)
    
    dataset_root:Path = args.data_root
    dataset = LP_Deblur_Dataset(
        data_root = dataset_root, blur_aug = args.blur_aug,
        extraction=args.txt_extract, 
        cache_name=args.txt_cached, cached_file=args.txt_cached
    )
    logger.info(f'The number of training pairs = {len(dataset)}')
    trainloader = DataLoader(
        dataset=dataset, batch_size=args.batch_size, 
        shuffle=True, num_workers=int(args.num_threads)
    )
    val_dataset = LP_Deblur_OCR_Valiation_Dataset.build_dataset(
        dataroot=args.val_data_root,
        label_file=args.label_file
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
        validator = LPD_OCR_ACC_Evaluator(ocr_preprocess=L_CLAHE, current_best=0.0)

    lpdgan = LPDGAN_Trainer(
        logger=logger, epochs_policy= {
            'n_epochs':args.n_epochs,
            'n_epochs_decay':args.n_epochs_decay,
            'lr_decay_iters':args.lr_decay_iters
        },
        pretrained_weights=args.pretrained_weights,
        checkpoint_dir=args.checkpoint_dir, 
        gan_mode=args.gan_mode,
        lr=args.lr, lr_policy=args.lr_policy, lambda_L1=args.lambda_L1, 
        ocr_percepual=args.ocr_perceptual,
        gpu_id=args.gpu_id,
        txt_recons_dim=dataset.txt_d,
        plate_loss =args.txt_loss
    )
    
    total_iters = 0
    save_flag = 0
    epoch_loss = None
    for epoch in range(1, args.n_epochs + args.n_epochs_decay + 1):
        
        epoch_start_time = time.time()
        lpdgan.update_learning_rate(logger=logger)
        bar = tqdm(trainloader)

        epoch_val_flag = (not args.save_iter) and epoch % args.save_epoch_freq == 0
        
        for data in bar:
            n_samples = len(data['A_paths'])
            total_iters += n_samples
            lpdgan.optimize_parameters(input_x=data)
            bar.set_postfix(ordered_dict={"iters":total_iters})
            iter_val_flag = (args.save_iter) and (total_iters // args.save_iter_freq > save_flag)
            
            if epoch_val_flag:
                # epoch validation policy 
                iter_loss = lpdgan.get_current_losses()
                if epoch_loss is None:
                    epoch_loss = iter_loss.copy()
                else:
                    for k in iter_loss:
                        epoch_loss[k] += iter_loss[k]

            if iter_val_flag:
                # iteration validation policy 
                iter_loss = lpdgan.get_current_losses()
                tensorboard_writer.add_scalars("Losses",iter_loss, total_iters)
                logger.info(f"iters:{total_iters}:{iter_loss}")
                
                evaluation_and_save(
                    lpdgan=lpdgan, val_loader=val_loader, eva=validator, 
                    save_dir = args.model_save_root if val_loader is not None \
                        else args.model_save_root /f'iter_{total_iters}',
                    iters=total_iters, board=tensorboard_writer,logger=logger,
                    baseline=baseline
                )
                
                save_flag = total_iters // args.save_iter_freq

        if epoch_val_flag:
            # epoch validation policy 
            for k in epoch_loss:
                epoch_loss[k] /= len(dataset)

            tensorboard_writer.add_scalars("Losses",epoch_loss, epoch)
            logger.info(f"epoch:{epoch}:{epoch_loss}")

            evaluation_and_save(
                lpdgan=lpdgan, val_loader=val_loader,
                eva=validator, 
                save_dir = args.model_save_root if val_loader is not None \
                    else args.model_save_root /f'epoch_{epoch}',
                iters=epoch, board=tensorboard_writer,logger=logger,
                baseline=baseline, epoch=epoch
            )
            epoch_loss = None
            gc.collect()
        
        logger.info(f'End of epoch {epoch} / {args.n_epochs + args.n_epochs_decay}\t Time Taken: {time.time() - epoch_start_time} sec')
        lpdgan.save_networks(save_dir = args.model_save_root/'latest')
        lpdgan.save_optimizers(save_dir = args.model_save_root/'latest')
        logger.info(f"Last step optimizers and schedulers are saved at { args.model_save_root/'latest'}")
    
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
    parser.add_argument("--n_epochs", type=int, default=100, help='number of epochs with the initial learning rate')
    parser.add_argument("--n_epochs_decay", type=int, default=200, help='number of epochs to linearly decay learning rate to zero')
    parser.add_argument("--lr_decay_iters", type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
    parser.add_argument('--lr_policy', type=str, default='linear',help='learning rate policy. [linear | step | plateau | cosine]')

    # save mode 
    parser.add_argument('--save_epoch_freq', type=int, default=5)
    parser.add_argument('--save_iter', action='store_true')
    parser.add_argument('--save_iter_freq', type=int, default=5000)
    parser.add_argument("--model_save_root", type=Path, default=LPDGAN_DEFALUT_CKPT_DIR)
    
    # load pretrained model
    parser.add_argument('--pretrained_weights', type=Path, default=None)
    parser.add_argument('--checkpoint_dir', type=Path, default=None, help='Using the optimizers and schedulers saved from previous work to keep training')
    # lr
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
    
    # GAN
    parser.add_argument('--gan_mode', type=str, default='wgangp')
    
    # Loss function
    parser.add_argument("--ocr_perceptual", action='store_false')
    parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
    parser.add_argument('--txt_loss', type=str, default='probl1')
    parser.add_argument('--print_freq', type=int, default=10400)

    args = parser.parse_args()
    if args.blur_aug == "all":
        args.blur_aug = [_.name for _ in args.data_root.iterdir() if _.name != "sharp" and _.is_dir()]
    
    # print(args)
    # _ = input("ok ?")
    main(args)
