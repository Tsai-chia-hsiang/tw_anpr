from typing import Optional, Literal
from pathlib import Path
import os
from logging import Logger
import functools
import torch.nn as nn
import numpy as np
import torch
from .models import SwinTransformer_Backbone, get_config_or, load_networks
from .models.networks import NLayerDiscriminator, \
    PixelDiscriminator, PerceptualLoss, GANLoss, \
    get_scheduler
from .data import tensor2img, Spatial_Pyramid_cv2
from .models.taming.modules.losses.lpips import OCR_CRAFT_LPIPS

_LPDGAN_DIR_ = Path(os.path.abspath(__file__)).parent
LPDGAN_DEFALUT_CKPT_DIR = _LPDGAN_DIR_/"checkpoints"

check_gpu_id = lambda gpu_id: gpu_id > -1 and torch.cuda.is_available() and gpu_id < torch.cuda.device_count()

class SwinTrans_G(nn.Module):
    
    def __init__(self, pretrained_ckpt:Optional[Path]=None, gpu_id:int=0, mode:Literal["train", "inference"]="train", on_size:tuple[int, int]=(224, 112), show_log:bool=True):
        
        super(SwinTrans_G, self).__init__()
        self.mode = mode
        self.device = torch.device(f'cuda:{gpu_id}') if check_gpu_id(gpu_id=gpu_id) else torch.device('cpu')
        config_su = get_config_or()
        self.netG = SwinTransformer_Backbone(config_su)
        self.model_name = 'G'
        load_networks(pretrained_ckpt=pretrained_ckpt, net=self.netG, show_log=show_log)
        self.netG.to(device=self.device)
        self.on_size = on_size
        self.inference_aug = Spatial_Pyramid_cv2(org_size=self.on_size) 
        
        match self.mode:
            case "train":
                self.train()
            case "inference":
                self.eval()

    def forward(self, x:dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:

        fake_B, fake_B1, fake_B2, fake_B3, plate1, plate2 = self.netG(
            x['A0'].to(self.device), x['A1'].to(self.device), x['A2'].to(self.device)
        )
        return fake_B, fake_B1, fake_B2, fake_B3, plate1, plate2
    
    @torch.no_grad()
    def inference(self, x:np.ndarray, to_cv2:bool=True) -> np.ndarray:

        fake_B, _ ,_ ,_ ,_, _ = self(self.inference_aug(img=x, map_key='A', L=3, to_batch=True))
        
        return tensor2img(input_image=fake_B[0], to_cv2=to_cv2)
    
    @torch.no_grad()
    def batch_inference(self, x:dict[str, torch.Tensor],to_cv2:bool=True) -> list[np.ndarray]:
        fake_B, _ ,_ ,_ ,_, _ = self(x)
        fake_B = fake_B.cpu()
        return [tensor2img(fake_Bi, to_cv2=to_cv2) for fake_Bi in fake_B]
        

class LPDGAN(nn.Module):
    
    def __init__(
            self, logger:Logger, netG:SwinTrans_G, pretrained_dir:Path ,epochs_policy:dict[str, str], 
            gan_mode:Literal['wgangp', 'vanilla']='wgangp',
            lr:float=0.002,  lr_policy:Literal['linear', 'step', 'plateau', 'cosine']='linear',
            input_channel:int=3, output_channel:int=3, ndf:int = 64,
            lambda_L1:float=100.0,
            perceptual_loss:Literal['perceptual', 'OCR_perceptual'] = "OCR_perceptual",
            
        ) -> None:
        super().__init__()
        self.netG = netG
        self.device = netG.device
        self.gan_mode = gan_mode
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.ndf = ndf
        self.metric = 0
        self.lr_policy = lr_policy
        self.lambda_L1 = lambda_L1

        self.perceptualLoss = PerceptualLoss().to(self.device)
        
        self.OCR_perceptualLoss = None
        match perceptual_loss:
            case 'perceptual':
                self.OCR_perceptualLoss = PerceptualLoss().to(self.device).eval()
            case 'OCR_perceptual':
                self.OCR_perceptualLoss = OCR_CRAFT_LPIPS().to(self.device).eval()
            case _ :
                raise KeyError(f"Not support {perceptual_loss} yet")
        
        self.netD = NLayerDiscriminator(
            self.input_channel + self.output_channel, 
            self.ndf, n_layers=3, 
            norm_layer=functools.partial(
                nn.BatchNorm2d, affine=True, track_running_stats=True
            )
        ).to(self.device)
        
        self.netD1 = NLayerDiscriminator(
            self.input_channel + self.output_channel, 
            self.ndf, n_layers=3, 
            norm_layer=functools.partial(
                nn.BatchNorm2d, affine=True, track_running_stats=True
            )
        ).to(self.device)
        
        self.netD2 = NLayerDiscriminator(
            self.input_channel + self.output_channel, 
            self.ndf, n_layers=3, 
            norm_layer=functools.partial(
                nn.BatchNorm2d, affine=True, track_running_stats=True
            )
        ).to(self.device)

        self.netD_smallblock = PixelDiscriminator(
            self.input_channel, self.ndf, 
            norm_layer=functools.partial(
                nn.BatchNorm2d, affine=True, track_running_stats=True
            )
        ).to(self.device)
        
        self.model_names = ['G', 'D', 'D_smallblock', 'D1', 'D2']
        if pretrained_dir is not None:
            if pretrained_dir.is_dir():
                self.load_nets(pretrained_dir=pretrained_dir)
            else:
                logger.info(f"using random init..")
        else:
            logger.info(f"using random init..")

        self.loss_names = ['G_GAN', 'G_L1', 'PlateNum_L1', 'D_GAN', 'P_loss', 'D_real', 'D_fake', 'D_s']
        self.criterionL1 = torch.nn.L1Loss()
        self.criterionGAN = GANLoss(gan_mode).to(self.device)
        self.criterionGAN_s = GANLoss('lsgan').to(self.device)

        self.optimizer_G = torch.optim.Adam(
            self.netG.parameters(), lr=lr, betas=(0.5, 0.999)
        )
        
        self.optimizer_D = torch.optim.Adam(
            self.netD.parameters(), lr=lr, betas=(0.5, 0.999)
        )

        self.optimizer_D_smallblock = torch.optim.Adam(
            self.netD_smallblock.parameters(), lr=lr, betas=(0.5, 0.999)
        )
        self.optimizers = [self.optimizer_G, self.optimizer_D, self.optimizer_D_smallblock]
    
        self.schedulers = [get_scheduler(optimizer, lr_policy=self.lr_policy, **epochs_policy) for optimizer in self.optimizers]

    def set_input(self, input:dict[str, torch.Tensor|tuple[str]]):
    
        self.real_A = input['A0'].to(self.device)
        self.real_A1 = input['A1'].to(self.device)
        self.real_A2 = input['A2'].to(self.device)
        self.image_paths = input['A_paths']
        self.real_B = input['B0'].to(self.device)
        self.real_B1 = input['B1'].to(self.device)
        self.real_B2 = input['B2'].to(self.device)
        self.real_B3 = input['B3'].to(self.device)
        self.plate_info = input['plate_info'].to(self.device)

    def forward(self):
        self.fake_B, self.fake_B1, self.fake_B2, self.fake_B3, \
            self.plate1, self.plate2 = self.netG(
                {'A0':self.real_A, 'A1':self.real_A1, 'A2':self.real_A2}
            )
        self.fake_B_split = torch.chunk(self.fake_B, 7, dim=3)
        self.real_B_split = torch.chunk(self.real_B, 7, dim=3)
        self.real_A_split = torch.chunk(self.real_A, 7, dim=3)

    def set_requires_grad(self, nets:list[nn.Module]|nn.Module, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

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

        self.loss_PlateNum_L1:torch.Tensor = (
            self.criterionL1(self.plate1, self.plate_info) + \
            self.criterionL1(self.plate2, self.plate_info)
        ) / 2 * 0.01

        self.loss_G = self.loss_G_GAN + self.loss_G_s + self.loss_G_L1 + self.loss_P_loss + 0.1 * self.loss_PlateNum_L1
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def cal_gp(self, fake_AB, real_AB):
        r = torch.rand(size=(real_AB.shape[0], 1, 1, 1))
        r = r.cuda()
        x = (r * real_AB + (1 - r) * fake_AB).requires_grad_(True)
        d = self.netD(x)
        fake = torch.ones_like(d)
        fake = fake.cuda()
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
        if logger is None:
            print(f'learning rate {old_lr:.7f} -> {lr:.7f}')
        else:
            logger.info(f'learning rate {old_lr:.7f} -> {lr:.7f}')
        
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

    def load_nets(self, pretrained_dir:Path, logger:Optional[Logger] = None):
        if logger is not None:
            logger.info(f"load from {pretrained_dir}")
            
        for name in self.model_names:
            if isinstance(name, str) and name != 'G':
                load_networks(
                    net=getattr(self, 'net' + name), 
                    pretrained_ckpt=pretrained_dir/f'net_{name}.pth'
                )
    
    def get_current_losses(self) -> dict[str, float]:
        errors_ret = {}
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))
        return errors_ret