from typing import Optional, Literal
from pathlib import Path
from tqdm import tqdm
import os
from logging import Logger
import functools
import copy
import torch.nn as nn
import numpy as np
import torch
from torch.utils.data import DataLoader
from .models import SwinTransformer_Backbone, MBOSys, get_config_or, load_networks
from .models.networks import NLayerDiscriminator, \
    PixelDiscriminator, PerceptualLoss, GANLoss, TXT_REC_LOSSES, \
    get_scheduler
from .data import tensor2img, Spatial_Pyramid_cv2
from .models.taming.modules.losses.lpips import OCR_CRAFT_LPIPS
from .logger import print_infomation
import gc
_LPDGAN_DIR_ = Path(os.path.abspath(__file__)).parent
LPDGAN_DEFALUT_CKPT_DIR = _LPDGAN_DIR_/"checkpoints"
import sys
sys.path.append(str(_LPDGAN_DIR_.parent/f"anpr"))
from anpr import LicensePlate_OCR, OCR_Evaluator

check_gpu_id = lambda gpu_id: gpu_id > -1 and torch.cuda.is_available() and gpu_id < torch.cuda.device_count()

class SwinTrans_G(nn.Module):
    
    def __init__(self, on_size:tuple[int, int]=(224, 112)):
        
        super(SwinTrans_G, self).__init__()

        config_su = get_config_or()
        self.on_size = on_size
        self.netG = SwinTransformer_Backbone(config_su, img_size=on_size[0])
        self.inference_aug:Spatial_Pyramid_cv2 = None 
        self.device:torch.device = torch.device('cpu')
    
    def to(self, device:torch.device, **kwargs):
        self.device = device
        return super().to(device=device, **kwargs)
        
    def forward(self, x:dict[str, torch.Tensor],on_train:bool=False) -> tuple[torch.Tensor, list[torch.Tensor]|None]:
        #y, [fake_B3, fake_B2, fake_B1] 
        return self.netG(
            x['A0'].to(self.device), x['A1'].to(self.device), x['A2'].to(self.device),
            layer_out = on_train
        )
    
    @torch.no_grad()
    def inference(self, x:np.ndarray, to_cv2:bool=True) -> np.ndarray:
        #fake_B, _ ,_ ,_ ,_, _
        if self.inference_aug is None:
            self.inference_aug = Spatial_Pyramid_cv2(org_size=self.on_size) 
        
        fake_B, _ = self(self.inference_aug(img=x, map_key='A', L=3, to_batch=True), on_train=False)
        return tensor2img(input_image=fake_B[0], to_cv2=to_cv2)
    
    @torch.no_grad()
    def batch_inference(self, x:dict[str, torch.Tensor],to_cv2:bool=True) -> list[np.ndarray]:
        fake_B, _ = self(x, on_train=False)
        fake_B = fake_B.cpu()
        return [tensor2img(fake_Bi, to_cv2=to_cv2) for fake_Bi in fake_B]
        

class LPDGAN_Trainer(nn.Module):
    
    def __init__(
            self, epochs_policy:dict[str, str], 
            gan_mode:Literal['wgangp', 'vanilla']='wgangp',
            pretrained_weights:Optional[Path] = None ,
            checkpoint_dir:Optional[Path]=None,
            lr:float=0.0002,  lr_policy:Literal['linear', 'step', 'plateau', 'cosine']='linear',
            input_channel:int=3, output_channel:int=3, ndf:int = 64,
            lambda_L1:float=100.0, plate_loss:Literal['probl1', 'kl']='probl1',
            ocr_percepual:bool=True,
            logger:Optional[Logger]=None, 
            gpu_id:str='0',
            on_size:tuple[int,int]=(224, 112),
            txt_recons_dim:tuple[int, int]=(55, 97)
        ) -> None:
        super().__init__()
        self.device_id = gpu_id
        if gpu_id == 'cuda':
            self.device = torch.device('cuda')
        else:
            self.device = torch.device(f'cuda:{gpu_id}') if check_gpu_id(gpu_id=int(gpu_id)) else torch.device('cpu')

        self.gan_mode = gan_mode
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.ndf = ndf
        self.metric = 0
        self.lr_policy = lr_policy
        self.lambda_L1 = lambda_L1
        self.plate_loss_w = 0.001 if plate_loss == "kl" else 0.01
        self.plate_loss = TXT_REC_LOSSES[plate_loss]

        #swin transformer
        self.netG = SwinTrans_G(on_size=on_size)
        
        #Multi-res output and text reconstruction modules
        self.netMBO = MBOSys(
            img_sizes = [(14, 28), (28, 56), (56, 112)],
            embed_dims = [384, 192, 96],
            txt_seq=txt_recons_dim
        )

        self.netD = NLayerDiscriminator(
            self.input_channel + self.output_channel, 
            self.ndf, n_layers=3, 
            norm_layer=functools.partial(
                nn.BatchNorm2d, affine=True, track_running_stats=True
            )
        )
        
        self.netD1 = NLayerDiscriminator(
            self.input_channel + self.output_channel, 
            self.ndf, n_layers=3, 
            norm_layer=functools.partial(
                nn.BatchNorm2d, affine=True, track_running_stats=True
            )
        )
        
        self.netD2 = NLayerDiscriminator(
            self.input_channel + self.output_channel, 
            self.ndf, n_layers=3, 
            norm_layer=functools.partial(
                nn.BatchNorm2d, affine=True, track_running_stats=True
            )
        )

        self.netD_smallblock = PixelDiscriminator(
            self.input_channel, self.ndf, 
            norm_layer=functools.partial(
                nn.BatchNorm2d, affine=True, track_running_stats=True
            )
        )
        
        self.model_names = ['G','MBO','D', 'D_smallblock', 'D1', 'D2']
        self.load_nets(target=self.model_names, pretrained_dir = pretrained_weights, logger=logger)
        self._net_to_device(target=self.model_names)
    
        self.loss_names = ['G_GAN', 'G_L1', 'PlateNum', 'D_GAN', 'P_loss', 'D_real', 'D_fake', 'D_s']
        self.criterionL1 = torch.nn.L1Loss()
        self.criterionGAN = GANLoss(gan_mode).to(self.device)
        self.criterionGAN_s = GANLoss('lsgan').to(self.device)
        self.perceptualLoss = PerceptualLoss().to(self.device)
        self.OCR_perceptualLoss = OCR_CRAFT_LPIPS().to(self.device).eval() if ocr_percepual else None
        

        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=lr, betas=(0.5, 0.999))        
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=lr, betas=(0.5, 0.999))
        self.optimizer_D_smallblock = torch.optim.Adam(self.netD_smallblock.parameters(), lr=lr, betas=(0.5, 0.999))

        self.optimizers = [self.optimizer_G, self.optimizer_D, self.optimizer_D_smallblock]    
        self.schedulers = [get_scheduler(optimizer, lr_policy=self.lr_policy, **epochs_policy) for optimizer in self.optimizers]
        self.load_opts_and_schedulers(checkpoint=checkpoint_dir, logger=logger)
        gc.collect()
        
    def _net_to_device(self, target:list[str]):
        for net_name in target:
            
            net:nn.Module = getattr(self, 'net' + net_name, None)
            if net is None:
                print(f"No {net_name} such a net, please check")
                continue

            net.to(device=self.device)
            if self.device_id == 'cuda':
                setattr(self, 'net' + net_name, nn.DataParallel(net))
    
    def set_requires_grad(self, nets:list[nn.Module]|nn.Module, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def forward(self, input_x:dict[str, torch.Tensor|tuple[str]]):
        
        self.real_A = input_x['A0'].to(self.device)
        self.real_A1 = input_x['A1'].to(self.device)
        self.real_A2 = input_x['A2'].to(self.device)
        self.image_paths = input_x['A_paths'] if 'A_paths' in input_x else None
        self.real_B = input_x['B0'].to(self.device)
        self.real_B1 = input_x['B1'].to(self.device)
        self.real_B2 = input_x['B2'].to(self.device)
        self.real_B3 = input_x['B3'].to(self.device)
        self.plate_info = input_x['plate_info'].to(self.device)

        self.fake_B, ret = self.netG(
            {'A0':self.real_A, 'A1':self.real_A1, 'A2':self.real_A2},
            on_train=True
        )
        
        self.fake_B3, _ = self.netMBO(ret[0], layer_idx=0)
        self.fake_B2, self.plate2 = self.netMBO(ret[1], layer_idx=1)
        self.fake_B1, self.plate1 = self.netMBO(ret[2], layer_idx=2)
        
        self.fake_B_split = torch.chunk(self.fake_B, 7, dim=3)
        self.real_B_split = torch.chunk(self.real_B, 7, dim=3)
        self.real_A_split = torch.chunk(self.real_A, 7, dim=3)

    #Loss 
    def cal_small_D(self) -> torch.Tensor:
        loss_D_s_fake = 0
        loss_D_s_real = 0
        for i in range(len(self.fake_B_split)):
            pred_s_fake = self.netD_smallblock(self.fake_B_split[i].detach())
            loss_D_s_fake_tmp = self.criterionGAN_s(pred_s_fake, False)

            pred_s_real = self.netD_smallblock(self.real_B_split[i].detach())
            loss_D_s_real_tmp = self.criterionGAN_s(pred_s_real, True)

            loss_D_s_fake += loss_D_s_fake_tmp
            loss_D_s_real += loss_D_s_real_tmp

        return loss_D_s_fake / 7.0, loss_D_s_real / 7.0

    def cal_small_G(self) -> torch.Tensor:
        loss_G_s_fake = 0
        for i in range(len(self.fake_B_split)):
            pred_s_fake = self.netD_smallblock(self.fake_B_split[i].detach())
            loss_G_s_fake_tmp = self.criterionGAN_s(pred_s_fake, True)

            loss_G_s_fake += loss_G_s_fake_tmp

        return loss_G_s_fake / 7.0

    def backward_D(self):
        fake_AB = torch.cat((self.real_A, self.fake_B),
                            1)
        pred_fake = self.netD(fake_AB.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)

        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        loss_D_real = self.criterionGAN(pred_real, True)

        fake_AB1 = torch.cat((self.real_A1, self.fake_B1),
                             1)
        pred_fake1 = self.netD1(fake_AB1.detach())
        loss_D_fake1 = self.criterionGAN(pred_fake1, False)

        real_AB1 = torch.cat((self.real_A1, self.real_B1), 1)
        pred_real1 = self.netD1(real_AB1)
        loss_D_real1 = self.criterionGAN(pred_real1, True)

        fake_AB2 = torch.cat((self.real_A2, self.fake_B2),
                             1)
        pred_fake2 = self.netD2(fake_AB2.detach())
        loss_D_fake2 = self.criterionGAN(pred_fake2, False)

        real_AB2 = torch.cat((self.real_A2, self.real_B2), 1)
        pred_real2 = self.netD2(real_AB2)
        loss_D_real2 = self.criterionGAN(pred_real2, True)

        self.loss_D_fake = (loss_D_fake + loss_D_fake1 + loss_D_fake2) / 3
        self.loss_D_real = (loss_D_real + loss_D_real1 + loss_D_real2) / 3

        self.loss_D_GAN = (loss_D_fake + loss_D_real + loss_D_fake1 + loss_D_real1 +
                           loss_D_fake2 + loss_D_real2) * 0.5 / 3

        loss_D_s_fake, loss_D_s_real = self.cal_small_D()
        self.loss_D_s = (loss_D_s_fake + loss_D_s_real) * 0.5

        self.loss_D_gp = (self.cal_gp(fake_AB, real_AB) + self.cal_gp(fake_AB1, real_AB1) +
                          self.cal_gp(fake_AB2, real_AB2)) * 10 / 3

        self.loss_D = self.loss_D_GAN + self.loss_D_gp + self.loss_D_s
        self.loss_D.backward(retain_graph=True)

    def backward_G(self):
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        loss_G_GAN = self.criterionGAN(pred_fake, True)

        fake_AB1 = torch.cat((self.real_A1, self.fake_B1), 1)
        pred_fake1 = self.netD1(fake_AB1)
        loss_G_GAN1 = self.criterionGAN(pred_fake1, True)

        fake_AB2 = torch.cat((self.real_A2, self.fake_B2), 1)
        pred_fake2 = self.netD2(fake_AB2)
        loss_G_GAN2 = self.criterionGAN(pred_fake2, True)

        self.loss_G_GAN = (loss_G_GAN + loss_G_GAN1 + loss_G_GAN2) / 3

        self.loss_G_s = self.cal_small_G()

        loss_G_L1:torch.Tensor = self.criterionL1(self.fake_B, self.real_B) * self.lambda_L1
        loss_G_L11:torch.Tensor = self.criterionL1(self.fake_B1, self.real_B1) * self.lambda_L1
        loss_G_L12 :torch.Tensor= self.criterionL1(self.fake_B2, self.real_B2) * self.lambda_L1
        loss_G_L13:torch.Tensor = self.criterionL1(self.fake_B3, self.real_B3) * self.lambda_L1

        self.loss_G_L1 = (loss_G_L1 + loss_G_L11 + loss_G_L12 + loss_G_L13) / 4 * 0.01

        if self.OCR_perceptualLoss is not None:
            loss_P_loss = self.OCR_perceptualLoss(self.fake_B, self.real_B)
            loss_P_loss1 = self.OCR_perceptualLoss(self.fake_B1, self.real_B1)
            loss_P_loss2 = self.OCR_perceptualLoss(self.fake_B2, self.real_B2)
        else:
            loss_P_loss = self.perceptualLoss(self.fake_B, self.real_B)
            loss_P_loss1 = self.perceptualLoss(self.fake_B1, self.real_B1)
            loss_P_loss2 = self.perceptualLoss(self.fake_B2, self.real_B2)
        
        loss_P_loss3:torch.Tensor = self.perceptualLoss(self.fake_B3, self.real_B3)

        self.loss_P_loss = (loss_P_loss + loss_P_loss1 + loss_P_loss2 + loss_P_loss3) / 4 * 0.01
        
        self.loss_PlateNum:torch.Tensor = (
            self.plate_loss(self.plate1, self.plate_info) + \
            self.plate_loss(self.plate2, self.plate_info)
        ) / 2 * self.plate_loss_w
       
        self.loss_G = self.loss_G_GAN + self.loss_G_s + self.loss_G_L1 + self.loss_P_loss + 0.1 * self.loss_PlateNum
        self.loss_G.backward()

    #enter point : optimize the paras
    def optimize_parameters(self, input_x:dict[str, torch.Tensor|tuple[str]]):
        self.forward(input_x=input_x)
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def cal_gp(self, fake_AB, real_AB):
        r = torch.rand(size=(real_AB.shape[0], 1, 1, 1), device=self.device)
        x = (r * real_AB + (1 - r) * fake_AB).requires_grad_(True)
        d = self.netD(x)
        fake = torch.ones_like(d, device=self.device)
        g = torch.autograd.grad(
            outputs=d,
            inputs=x,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True
        )[0]
        gp = ((g.norm(2, dim=1) - 1) ** 2).mean()
        return gp

    def get_image_paths(self):
        return self.image_paths

    def update_learning_rate(self, logger:Optional[Logger]=None):
        old_lr = self.optimizers[0].param_groups[0]['lr']
        for scheduler in self.schedulers:
            if self.lr_policy == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr']
        print_infomation(f'learning rate {old_lr:.7f} -> {lr:.7f}', logger=logger)
        
    def get_current_losses(self) -> dict[str, float]:
        errors_ret = {}
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))
        return errors_ret
    
    ## net load & save
    def save_networks(self, save_dir:Path):
        save_dir.mkdir(parents=True, exist_ok=True)
        for name in self.model_names:
            if isinstance(name, str):
                save_path = save_dir/f'net_{name}.pth'
                net = getattr(self, 'net' + name)
                if isinstance(net, nn.DataParallel):
                    torch.save(net.module.state_dict(), save_path)  # Access .module if wrapped in DataParallel
                else:
                    torch.save(net.state_dict(), save_path)  # Directly save if not wrapped in DataParallel

    def load_nets(self, target:list[str], pretrained_dir:Optional[Path]=None, logger:Optional[Logger] = None):
        
        if pretrained_dir is None:
            print_infomation("Will use random init ..", logger=logger)
            return
        
        if not pretrained_dir.is_dir():
            print_infomation(f"No {pretrained_dir} such a directory, will use random init ..", logger=logger)
            return 
        
        print_infomation(f"load from {pretrained_dir}", logger=logger)    
        
        for name in target:
            net = getattr(self, 'net' + name, None)
            if net is None:
                print_infomation(f"No {name} such a net to load, please check", logger=logger)
                continue
        
            load_networks(net=net, pretrained_ckpt=pretrained_dir/f'net_{name}.pth', logger=logger)
    
    ## optimizers & schedulers load & save
    def save_optimizers(self, save_dir:Path):
        """
        save optimizers and schedulers
        """
        save_dir.mkdir(parents=True, exist_ok=True)
        for opt_name, opt, sched in zip(["G", "D", "D_small"], self.optimizers, self.schedulers):
            torch.save(
                opt.state_dict(),
                save_dir/f"{opt_name}_opt.pth"
            )
            torch.save(
                sched.state_dict(),
                save_dir/f"{opt_name}_sche.pth"
            )
        
    def load_opts_and_schedulers(self, checkpoint:Optional[Path]=None, logger:Optional[Logger]=None):
        
        if checkpoint is None:
            print_infomation("using init status", logger=logger)
            return 
        
        if not checkpoint.is_dir():
            print_infomation(f"No such {checkpoint}, using init status")
            return 
        
        for opt_name, opt, sched in zip(["G", "D", "D_small"], self.optimizers, self.schedulers): 
            print_infomation(
                f"{opt_name}: {checkpoint/f'{opt_name}_opt.pth'} & {checkpoint/f'{opt_name}_sche.pth'}", 
                logger=logger
            )
            load_networks(opt, checkpoint/f"{opt_name}_opt.pth", logger=logger)
            load_networks(sched, checkpoint/f"{opt_name}_sche.pth", logger=logger)


class LPD_OCR_ACC_Evaluator(OCR_Evaluator):
    
    update_signal = "update"
    keep_signal = "keep"
    
    def __init__(self, ocr_preprocess, current_best=0) -> None:
        super().__init__()
        self.ocr = LicensePlate_OCR()
        self.preprocess = ocr_preprocess
        self._current_best = {
            'cer':current_best,
            'lcs':current_best
        }
        self.hist = {
            'cer':[],
            'lcs':[]
        }
    @property
    def current_best(self):
        return copy.deepcopy(self._current_best)
        
    def val_LP_db_dataset(self, val_loader:DataLoader, swintrans_g:SwinTrans_G) -> dict[str, float]:
        pred = []
        gth = []
        for data in tqdm(val_loader):
            imgs = swintrans_g.batch_inference(x=data)
            for img in imgs:
                img = self.preprocess(img)
                prediction = self.ocr(img)
                pred.append(prediction[0])  
            gth += list(data['gth']) 
            
        acc_cer = self(pred, gth, 'cer', to_acc=True)
        acc_lcs = self(pred, gth, 'lcs')
        return {'cer':acc_cer, 'lcs':acc_lcs}

    def update(self, accs:dict[str, float])->dict[str, str]:
        ret={'cer':None, 'lcs':None}
        
        for k, v in accs.items():
            self.hist[k].append(v)
            if v >= self._current_best[k]:
                ret[k] =  LPD_OCR_ACC_Evaluator.update_signal
                self._current_best[k] = v
            else:
                ret[k] = LPD_OCR_ACC_Evaluator.keep_signal
        
        return ret
