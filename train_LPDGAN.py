"""
Refactoring from https://github.com/haoyGONG/LPDGAN.git
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import json
import time
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from torch.utils.data import DataLoader
from pathlib import Path
from logging import Logger
from LPDGAN import LP_Deblur_Dataset, LP_Deblur_OCR_Valiation_Dataset,\
    SwinTrans_G, LPDGAN_Trainer, LPDGAN_DEFALUT_CKPT_DIR
from LPDGAN.logger import get_logger, remove_old_tf_evenfile
from anpr.ocr import cer, LicensePlate_OCR

ocr_model = LicensePlate_OCR()

def ocr_validation(db_model:SwinTrans_G, val_loader:LP_Deblur_OCR_Valiation_Dataset) -> float:
    pred = []
    gth = []
    db_model.eval()
    for data in val_loader:
        imgs = db_model.batch_inference(x=data)
        for img in imgs:
            txt = ocr_model(img)[0]
            if txt == 'n':
                txt = ''
            pred.append(txt)  
        gth += list(data['gth']) 
    acc = cer(pred=pred, gth=gth, to_acc=True)
    return acc

def main(args:Namespace):
    
    args.model_save_root.mkdir(parents=True, exist_ok=True)
    logger = get_logger(name=__name__, file=args.model_save_root/"training.log")
    
    logger.info(f"{args}")
    
    dataset_root:Path = args.data_root
    dataset = LP_Deblur_Dataset(data_root = dataset_root, mode='train', blur_aug = args.blur_aug)
    logger.info(f'The number of training pairs = {len(dataset)}')

    val_dataset = LP_Deblur_OCR_Valiation_Dataset.build_dataset(
        dataroot=args.val_data_root,
        label_file=args.label_file
    )
    val_loader = None
    if val_dataset is not None:
        logger.info(f"validation set : {args.val_data_root}, label:{args.label_file}")
        logger.info(f"Validation number : {len(val_dataset)}")
        val_loader = DataLoader(
            dataset=val_dataset, batch_size=args.batch_size, 
            shuffle=False
        )
    trainloader = DataLoader(
        dataset=dataset, batch_size=args.batch_size, 
        shuffle=True, num_workers=int(args.num_threads)
    )
    
    pretrained_ckpt=args.pretrained_dir/f"net_G.pth" if args.pretrained_dir is not None else None
    #Path("LPDGAN/checkpoints/diff_blur/190_net_G.pth")
    Swin_Generator = SwinTrans_G(
        pretrained_ckpt=pretrained_ckpt,
        gpu_id=args.gpu_id, mode='train', on_size=(224, 112)
    )

    lpdgan = LPDGAN_Trainer(
        netG=Swin_Generator, logger=logger, pretrained_dir=args.pretrained_dir,  
        gan_mode=args.gan_mode, epochs_policy= {
            'n_epochs':args.n_epochs,
            'n_epochs_decay':args.n_epochs_decay,
            'lr_decay_iters':args.lr_decay_iters
        },
        lr=args.lr, lr_policy=args.lr_policy, lambda_L1=args.lambda_L1, 
        perceptual_loss=args.perceptual
    )
    
    total_iters = 0
    acc = 0
    for epoch in range(1, args.n_epochs + args.n_epochs_decay + 1):
        epoch_start_time = time.time()
        epoch_iter = 0
        
        lpdgan.update_learning_rate(logger=logger)
        for data in tqdm(trainloader):
            
            total_iters += args.batch_size
            epoch_iter += args.batch_size
            lpdgan.set_input(data)
            lpdgan.optimize_parameters()

            if total_iters % args.print_freq == 0:
                logger.info(f"loss {total_iters} : {lpdgan.get_current_losses()}")

            if total_iters % args.save_latest_freq == 0:
                if val_loader is None:
                    logger.info(f"saving the latest model (epoch {epoch} total_iters {total_iters})")
                    prefix = f'iter_{total_iters}' if args.save_by_iter else 'latest'
                    lpdgan.save_networks(save_dir = args.model_save_root/prefix)
                else:
                    logger.info(f"validation at {total_iters}")
                    acc_iter = ocr_validation(db_model= Swin_Generator,val_loader=val_loader)
                    logger.info(f"OCR Accuracy : {acc_iter}")
                    if acc_iter >= acc:
                        logger.info("update")
                        acc = acc_iter
                        lpdgan.save_networks(save_dir=args.model_save_root/"ocr_best")
                    
                    Swin_Generator.train()

        if epoch % args.save_epoch_freq == 0:
            if val_loader is None:
                logger.info(f"saving the model at the end of epoch {epoch}, iters {total_iters}")
                lpdgan.save_networks( args.model_save_root/'latest')
                lpdgan.save_networks( args.model_save_root/f'epoch_{epoch}')
            else:
                logger.info(f"validation at epoch {epoch}")
                acc_iter = ocr_validation(db_model= Swin_Generator,val_loader=val_loader)
                logger.info(f"OCR Accuracy : {acc_iter}")
                if acc_iter >= acc:
                    logger.info("update")
                    acc = acc_iter
                    lpdgan.save_networks(save_dir=args.model_save_root/"ocr_best")
                Swin_Generator.train()

        logger.info(f'End of epoch {epoch} / {args.n_epochs + args.n_epochs_decay}\t Time Taken: {time.time() - epoch_start_time} sec')



if __name__ == "__main__":
    
    parser = ArgumentParser()
    # Data path
    parser.add_argument("--data_root", type=Path, default=Path("./dataset")/"tw_post"/"train")
    parser.add_argument("--blur_aug", nargs='+', type=str, default='all')

    parser.add_argument("--val_data_root", type=Path, default=None)
    parser.add_argument("--label_file", type=Path, default=None)

    parser.add_argument('--num_threads', default=0, type=int, help='# threads for loading data')
    parser.add_argument("--gpu_id", type=int, default=0)

    # Batch size & epochs & lr scheduler
    parser.add_argument("--batch_size", type=int, default=15)
    parser.add_argument("--n_epochs", type=int, default=100, help='number of epochs with the initial learning rate')
    parser.add_argument("--n_epochs_decay", type=int, default=100, help='number of epochs to linearly decay learning rate to zero')
    parser.add_argument("--lr_decay_iters", type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
    parser.add_argument('--lr_policy', type=str, default='linear',
                        help='learning rate policy. [linear | step | plateau | cosine]')

    # save mode 
    parser.add_argument('--save_by_iter', action='store_true')
    parser.add_argument('--save_epoch_freq', type=int, default=5)
    parser.add_argument('--save_latest_freq', type=int, default=5000)
    parser.add_argument("--model_save_root", type=Path, default=LPDGAN_DEFALUT_CKPT_DIR)
    
    # load pretrained model
    parser.add_argument('--pretrained_dir', type=Path, default=None)
    parser.add_argument('--load_iter', type=int, default=200)

    # lr
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
    
    # GAN
    parser.add_argument('--gan_mode', type=str, default='wgangp')
    
    # Loss function
    parser.add_argument("--perceptual", type=str, default="OCR_perceptual")
    parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
    parser.add_argument('--print_freq', type=int, default=10400)

    args = parser.parse_args()
    if args.blur_aug == "all":
        args.blur_aug = [_.name for _ in args.data_root.iterdir() if _.name != "sharp"]
    
    main(args)