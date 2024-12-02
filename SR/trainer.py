import torch
import os
from collections import OrderedDict
from torch.nn import functional as F
from torch.nn.functional import l1_loss
from torchvision.models.vgg import VGG19_Weights
from torch import nn as nn
from torchvision.models import vgg as vgg
from .network import RRDBNet, UNetDiscriminatorSN
from .import SR_PRETRAINED_DIR, SR_FILE_DIR
import sys 
sys.path.append(os.path.abspath(SR_FILE_DIR.parent))
from pytorchocr.OCR_percetualloss import TextRec_Loss

VGG_PRETRAIN_PATH = str(SR_PRETRAINED_DIR/f"vgg19-dcbb9e9d.pth")
NAMES = {
    'vgg11': [
        'conv1_1', 'relu1_1', 'pool1', 'conv2_1', 'relu2_1', 'pool2', 'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2',
        'pool3', 'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'pool4', 'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2',
        'pool5'
    ],
    'vgg13': [
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'pool3', 'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'pool4',
        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'pool5'
    ],
    'vgg16': [
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'pool3', 'conv4_1', 'relu4_1', 'conv4_2',
        'relu4_2', 'conv4_3', 'relu4_3', 'pool4', 'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3',
        'pool5'
    ],
    'vgg19': [
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3', 'conv4_1',
        'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4', 'conv5_1', 'relu5_1',
        'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5'
    ]
}


def insert_bn(names):
    """Insert bn layer after each conv.

    Args:
        names (list): The list of layer names.

    Returns:
        list: The list of layer names with bn layers.
    """
    names_bn = []
    for name in names:
        names_bn.append(name)
        if 'conv' in name:
            position = name.replace('conv', '')
            names_bn.append('bn' + position)
    return names_bn


class VGGFeatureExtractor(nn.Module):
    """VGG network for feature extraction.

    In this implementation, we allow users to choose whether use normalization
    in the input feature and the type of vgg network. Note that the pretrained
    path must fit the vgg type.

    Args:
        layer_name_list (list[str]): Forward function returns the corresponding
            features according to the layer_name_list.
            Example: {'relu1_1', 'relu2_1', 'relu3_1'}.
        vgg_type (str): Set the type of vgg network. Default: 'vgg19'.
        use_input_norm (bool): If True, normalize the input image. Importantly,
            the input feature must in the range [0, 1]. Default: True.
        range_norm (bool): If True, norm images with range [-1, 1] to [0, 1].
            Default: False.
        requires_grad (bool): If true, the parameters of VGG network will be
            optimized. Default: False.
        remove_pooling (bool): If true, the max pooling operations in VGG net
            will be removed. Default: False.
        pooling_stride (int): The stride of max pooling operation. Default: 2.
    """

    def __init__(self,
                 layer_name_list,
                 vgg_type='vgg19',
                 use_input_norm=True,
                 range_norm=False,
                 requires_grad=False,
                 remove_pooling=False,
                 pooling_stride=2):
        super(VGGFeatureExtractor, self).__init__()

        self.layer_name_list = layer_name_list
        self.use_input_norm = use_input_norm
        self.range_norm = range_norm

        self.names = NAMES[vgg_type.replace('_bn', '')]
        if 'bn' in vgg_type:
            self.names = insert_bn(self.names)

        # only borrow layers that will be used to avoid unused params
        max_idx = 0
        for v in layer_name_list:
            idx = self.names.index(v)
            if idx > max_idx:
                max_idx = idx

        if os.path.exists(VGG_PRETRAIN_PATH):
            vgg_net = getattr(vgg, vgg_type)(pretrained=False)
            state_dict = torch.load(VGG_PRETRAIN_PATH, map_location=lambda storage, loc: storage)
            vgg_net.load_state_dict(state_dict)
        else:
            vgg_net = getattr(vgg, vgg_type)(weights=VGG19_Weights.DEFAULT)

        features = vgg_net.features[:max_idx + 1]

        modified_net = OrderedDict()
        for k, v in zip(self.names, features):
            if 'pool' in k:
                # if remove_pooling is true, pooling operation will be removed
                if remove_pooling:
                    continue
                else:
                    # in some cases, we may want to change the default stride
                    modified_net[k] = nn.MaxPool2d(kernel_size=2, stride=pooling_stride)
            else:
                modified_net[k] = v

        self.vgg_net = nn.Sequential(modified_net)

        if not requires_grad:
            self.vgg_net.eval()
            for param in self.parameters():
                param.requires_grad = False
        else:
            self.vgg_net.train()
            for param in self.parameters():
                param.requires_grad = True

        if self.use_input_norm:
            # the mean is for image with range [0, 1]
            self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            # the std is for image with range [0, 1]
            self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        if self.range_norm:
            x = (x + 1) / 2
        if self.use_input_norm:
            x = (x - self.mean) / self.std

        output = {}
        for key, layer in self.vgg_net._modules.items():
            x = layer(x)
            if key in self.layer_name_list:
                output[key] = x.clone()

        return output


class PerceptualLoss(torch.nn.Module):
    """Perceptual loss with commonly used style loss.

    Args:
        layer_weights (dict): The weight for each layer of vgg feature.
            Here is an example: {'conv5_4': 1.}, which means the conv5_4
            feature layer (before relu5_4) will be extracted with weight
            1.0 in calculating losses.
        vgg_type (str): The type of vgg network used as feature extractor.
            Default: 'vgg19'.
        use_input_norm (bool):  If True, normalize the input image in vgg.
            Default: True.
        range_norm (bool): If True, norm images with range [-1, 1] to [0, 1].
            Default: False.
        perceptual_weight (float): If `perceptual_weight > 0`, the perceptual
            loss will be calculated and the loss will multiplied by the
            weight. Default: 1.0.
        style_weight (float): If `style_weight > 0`, the style loss will be
            calculated and the loss will multiplied by the weight.
            Default: 0.
        criterion (str): Criterion used for perceptual loss. Default: 'l1'.
    """

    def __init__(self,
                 layer_weights,
                 vgg_type='vgg19',
                 use_input_norm=True,
                 range_norm=False,
                 perceptual_weight=1.0,
                 style_weight=0.,
                 criterion='l1'):
        super(PerceptualLoss, self).__init__()
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.layer_weights = layer_weights
        self.vgg = VGGFeatureExtractor(
            layer_name_list=list(layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm,
            range_norm=range_norm)

        self.criterion_type = criterion
        if self.criterion_type == 'l1':
            self.criterion = torch.nn.L1Loss(reduction='mean')
        elif self.criterion_type == 'l2':
            self.criterion = torch.nn.MSELoss()
        elif self.criterion_type == 'fro':
            self.criterion = None
        else:
            raise NotImplementedError(f'{criterion} criterion has not been supported.')

    def forward(self, x, gt):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        # extract vgg features
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())

        # calculate perceptual loss
        if self.perceptual_weight > 0:
            percep_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    percep_loss += torch.norm(x_features[k] - gt_features[k], p='fro') * self.layer_weights[k]
                else:
                    percep_loss += self.criterion(x_features[k], gt_features[k]) * self.layer_weights[k]
            percep_loss *= self.perceptual_weight
        else:
            percep_loss = None

        # calculate style loss
        if self.style_weight > 0:
            style_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    style_loss += torch.norm(
                        self._gram_mat(x_features[k]) - self._gram_mat(gt_features[k]), p='fro') * self.layer_weights[k]
                else:
                    style_loss += self.criterion(self._gram_mat(x_features[k]), self._gram_mat(
                        gt_features[k])) * self.layer_weights[k]
            style_loss *= self.style_weight
        else:
            style_loss = None

        return percep_loss, style_loss

    def _gram_mat(self, x):
        """Calculate Gram matrix.

        Args:
            x (torch.Tensor): Tensor with shape of (n, c, h, w).

        Returns:
            torch.Tensor: Gram matrix.
        """
        n, c, h, w = x.size()
        features = x.view(n, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram

class GANLoss(torch.nn.Module):
    """Define GAN loss.

    Args:
        gan_type (str): Support 'vanilla', 'lsgan', 'wgan', 'hinge'.
        real_label_val (float): The value for real label. Default: 1.0.
        fake_label_val (float): The value for fake label. Default: 0.0.
        loss_weight (float): Loss weight. Default: 1.0.
            Note that loss_weight is only for generators; and it is always 1.0
            for discriminators.
    """

    def __init__(self, gan_type="vanilla", real_label_val=1.0, fake_label_val=0.0, loss_weight=1.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type
        self.loss_weight = loss_weight
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'vanilla':
            self.loss = torch.nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = torch.nn.MSELoss()
        elif self.gan_type == 'wgan':
            self.loss = self._wgan_loss
        elif self.gan_type == 'wgan_softplus':
            self.loss = self._wgan_softplus_loss
        elif self.gan_type == 'hinge':
            self.loss = torch.nn.ReLU()
        else:
            raise NotImplementedError(f'GAN type {self.gan_type} is not implemented.')

    def _wgan_loss(self, input:torch.Tensor, target:torch.Tensor):
        """wgan loss.

        Args:
            input (Tensor): Input tensor.
            target (bool): Target label.

        Returns:
            Tensor: wgan loss.
        """
        return -input.mean() if target else input.mean()

    def _wgan_softplus_loss(self, input, target):
        """wgan loss with soft plus. softplus is a smooth approximation to the
        ReLU function.

        In StyleGAN2, it is called:
            Logistic loss for discriminator;
            Non-saturating loss for generator.

        Args:
            input (Tensor): Input tensor.
            target (bool): Target label.

        Returns:
            Tensor: wgan loss.
        """
        return F.softplus(-input).mean() if target else F.softplus(input).mean()

    def get_target_label(self, input:torch.Tensor, target_is_real):
        """Get target label.

        Args:
            input (Tensor): Input tensor.
            target_is_real (bool): Whether the target is real or fake.

        Returns:
            (bool | Tensor): Target tensor. Return bool for wgan, otherwise,
                return Tensor.
        """

        if self.gan_type in ['wgan', 'wgan_softplus']:
            return target_is_real
        target_val = (self.real_label_val if target_is_real else self.fake_label_val)
        return input.new_ones(input.size()) * target_val

    def forward(self, input:torch.Tensor, target_is_real:bool, is_disc=False):
        """
        Args:
            input (Tensor): The input for the loss module, i.e., the network
                prediction.
            target_is_real (bool): Whether the targe is real or fake.
            is_disc (bool): Whether the loss for discriminators or not.
                Default: False.

        Returns:
            Tensor: GAN loss value.
        """
        target_label = self.get_target_label(input, target_is_real)
        if self.gan_type == 'hinge':
            if is_disc:  # for discriminators in hinge-gan
                input = -input if target_is_real else input
                loss = self.loss(1 + input).mean()
            else:  # for generators in hinge-gan
                loss = -input.mean()
        else:  # other gan types
            loss = self.loss(input, target_label)

        # loss_weight is always 1.0 for discriminators
        return loss if is_disc else loss * self.loss_weight


class ESRGANModel():
    """ESRGAN model for single image super-resolution."""

    def __init__(
            self, net_g:RRDBNet, net_d:UNetDiscriminatorSN, dev:torch.device,
            lr:float=1e-4, MultiStepLR_mileston:list[int]=[100], gamma:float=0.5,
            start_epoch:int=0,
            g_loss_w:float=1e-1, p_loss_w:float=1.0
        ):
        super(ESRGANModel, self).__init__()
        
        self.dev = dev
        self.cri_perceptual = PerceptualLoss(
            layer_weights={
                'conv1_2': 0.1,
                'conv2_2': 0.1,
                'conv3_4': 1,
                'conv4_4': 1,
                'conv5_4': 1
            },
            perceptual_weight=p_loss_w
        )
        self.current_lr = lr
        self.cri_gan = GANLoss(loss_weight=g_loss_w)
        
        self.ocr_per = TextRec_Loss(input_hw=(56, 112))

        self.net_g = net_g
        self.net_g.to(self.dev)
        self.net_d = net_d
        self.net_d.to(self.dev)
        self.optimizer_g = torch.optim.Adam(net_g.parameters(), lr=lr)
        self.optimizer_d = torch.optim.Adam(net_d.parameters(), lr=lr)
        self.scheduler_g = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer_g,
            gamma=gamma,
            milestones=MultiStepLR_mileston,  # Example milestones
            last_epoch=start_epoch - 1
        )
        self.scheduler_d = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer_d,
            milestones=MultiStepLR_mileston,  # Example milestones
            gamma=gamma,
            last_epoch=start_epoch - 1
        )
        self.cri_perceptual.to(self.dev)
        self.cri_gan.to(self.dev)
        self.ocr_per.to(self.dev)

    def optimize_parameters(self, lq:torch.Tensor, gt:torch.Tensor, train=True)-> dict[str, float]:
        # optimize net_g
        if train:
            self.net_d.train()
            self.net_g.train()
        else:
            self.net_d.eval()
            self.net_g.eval()
        
        lq = lq.to(self.dev)
        gt = gt.to(self.dev)
        for p in self.net_d.parameters():
            p.requires_grad = False
        
        if train:
            self.optimizer_g.zero_grad()
        
        output = self.net_g(lq)

        l_g_total = 0
        loss_dict = {}

        l_g_pix = l1_loss(output, gt, reduction='mean')
        l_g_total += l_g_pix
        loss_dict['l_g_pix'] = l_g_pix.item()
        
        # perceptual loss
        l_g_percep, l_g_style = self.cri_perceptual(output, gt)
        
        if l_g_percep is not None:
            l_g_total += l_g_percep
            loss_dict['l_g_percep'] = l_g_percep.item()
        
        if l_g_style is not None:
            l_g_total += l_g_style
            loss_dict['l_g_style'] = l_g_style.item()
        
        l_g_ocr_p = self.ocr_per(output, gt)
        l_g_total += l_g_ocr_p
        
        loss_dict['l_g_ocr_p'] = l_g_ocr_p.item()

        # gan loss (relativistic gan)
        real_d_pred = self.net_d(gt).detach()
        fake_g_pred = self.net_d(output)
        l_g_real = self.cri_gan(real_d_pred - torch.mean(fake_g_pred), False, is_disc=False)
        l_g_fake = self.cri_gan(fake_g_pred - torch.mean(real_d_pred), True, is_disc=False)
        l_g_gan = (l_g_real + l_g_fake) / 2

        l_g_total += l_g_gan
        loss_dict['l_g_gan'] = l_g_gan.item()
        
        if train:
            l_g_total.backward()
            self.optimizer_g.step()

        # optimize net_d
        for p in self.net_d.parameters():
            p.requires_grad = True
        
        if train:
            self.optimizer_d.zero_grad()
        # gan loss (relativistic gan)

        # In order to avoid the error in distributed training:
        # "Error detected in CudnnBatchNormBackward: RuntimeError: one of
        # the variables needed for gradient computation has been modified by
        # an inplace operation",
        # we separate the backwards for real and fake, and also detach the
        # tensor for calculating mean.

        # real
        fake_d_pred = self.net_d(output).detach()
        real_d_pred = self.net_d(gt)
        l_d_real = self.cri_gan(real_d_pred - torch.mean(fake_d_pred), True, is_disc=True) * 0.5
        if train:
            l_d_real.backward()
        # fake
        fake_d_pred = self.net_d(output.detach())
        l_d_fake = self.cri_gan(fake_d_pred - torch.mean(real_d_pred.detach()), False, is_disc=True) * 0.5
        
        if train:
            l_d_fake.backward()
            self.optimizer_d.step()

        loss_dict['l_d_real'] = l_d_real.item()
        loss_dict['l_d_fake'] = l_d_fake.item()
        loss_dict['out_d_real'] = torch.mean(real_d_pred.detach()).item()
        loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach()).item()

        return loss_dict
    
    def update_lr(self) -> float:
        self.scheduler_g.step()
        self.scheduler_d.step()
        self.current_lr = self.optimizer_g.param_groups[0]['lr']
        
    
if __name__ == "__main__":
    pl = PerceptualLoss(
            layer_weights={
                'conv1_2': 0.1,
                'conv2_2': 0.1,
                'conv3_4': 1,
                'conv4_4': 1,
                'conv5_4': 1
            }
        )