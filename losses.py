import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from basicsr.losses.basic_loss import PerceptualLoss

# from kornia.constants import pi
_reduction_modes = ['none', 'mean', 'sum']


def tv_loss(x, beta=0.5, reg_coeff=5):
    '''Calculates TV loss for an image `x`.
        
    Args:
        x: image, torch.Variable of torch.Tensor
        beta: See https://arxiv.org/abs/1412.0035 (fig. 2) to see effect of `beta` 
    '''
    dh = torch.pow(x[:, :, :, 1:] - x[:, :, :, :-1], 2)
    dw = torch.pow(x[:, :, 1:, :] - x[:, :, :-1, :], 2)
    a, b, c, d = x.shape
    return reg_coeff * (torch.sum(torch.pow(dh[:, :, :-1] + dw[:, :, :, :-1], beta)) / (a * b * c * d))


class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)

    Args:
        eps: Small constant to avoid division by zero.
    """

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps * self.eps)))  
        return loss


class CharFreqLoss(nn.Module):
    """L1 (mean absolute error, MAE) loss of fft.
    
    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(CharFreqLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        # self.reduction = reduction
        # self.l1_loss = CharbonnierLoss(loss_weight, reduction)
        self.l1_loss = CharbonnierLoss()

    def forward(self, pred, target):
        diff = torch.fft.rfft2(pred) - torch.fft.rfft2(target)
        loss = torch.mean(torch.abs(diff))
        # print(loss)
        return self.loss_weight * loss * 0.01 + self.l1_loss(pred, target)


class CharFreqPerceptualLoss(nn.Module):
    """L1 (mean absolute error, MAE) loss of fft.
    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    Return:
        Loss tensor with the same shape as input
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(CharFreqPerceptualLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight

        self.l1_loss = CharbonnierLoss()
        self.perceptual_loss = PerceptualLoss(layer_weights={'conv5_4': 1.0}, perceptual_weight=1.0)

    def forward(self, pred, target):
        diff = torch.fft.rfft2(pred) - torch.fft.rfft2(target)
        fft_loss = torch.mean(torch.abs(diff))
        perceptual_loss, _ = self.perceptual_loss(pred, target)

        return fft_loss * 0.01 + self.l1_loss(pred, target) + perceptual_loss*0.05

class CharFreqPerceptualLossAblation(nn.Module):
    """L1 (mean absolute error, MAE) loss of fft.
    
    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    Return:
        Loss tensor with the same shape as input
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(CharFreqPerceptualLossAblation, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight

        self.l1_loss = CharbonnierLoss()
        self.perceptual_loss = PerceptualLoss(layer_weights={'conv5_4': 1.0}, perceptual_weight=1.0)

    def forward(self, pred, target):
        diff = torch.fft.rfft2(pred) - torch.fft.rfft2(target)
        fft_loss = torch.mean(torch.abs(diff))
        perceptual_loss, _ = self.perceptual_loss(pred, target)

        return fft_loss * 0.01 + self.l1_loss(pred, target) + perceptual_loss*0.001

class CharPerceptualLoss(nn.Module):
    """L1 (mean absolute error, MAE) loss of fft.
    
    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    Return:
        Loss tensor with the same shape as input
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(CharPerceptualLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight

        self.l1_loss = CharbonnierLoss()
        self.perceptual_loss = PerceptualLoss(layer_weights={'conv5_4': 1.0}, perceptual_weight=1.0)

    def forward(self, pred, target):

        perceptual_loss, _ = self.perceptual_loss(pred, target)

        return self.l1_loss(pred, target) + perceptual_loss*0.05

class CharFreqPerceptualLossV2(nn.Module):
    """L1 (mean absolute error, MAE) loss of fft.
    
    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    Return:
        Loss tensor with the same shape as input
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(CharFreqPerceptualLossV2, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight

        self.l1_loss = CharbonnierLoss()
        self.perceptual_loss = PerceptualLoss(layer_weights={'conv5_4': 1.0}, perceptual_weight=1.0)

    def forward(self, pred, target):
        diff = torch.fft.rfft2(pred) - torch.fft.rfft2(target)  
        fft_loss = torch.mean(torch.abs(diff))
        perceptual_loss, _ = self.perceptual_loss(pred, target)

        pix_loss = self.l1_loss(pred, target)
        frequency_loss = fft_loss * 0.01
        perceptual_loss = perceptual_loss * 0.05

        total_loss = pix_loss + frequency_loss + perceptual_loss

        return total_loss, pix_loss, frequency_loss, perceptual_loss

class Stripformer_Loss(nn.Module):

    def __init__(self, ):
        super(Stripformer_Loss, self).__init__()

        self.char = CharbonnierLoss()
        self.edge = EdgeLoss()
        self.contrastive = ContrastLoss()

    def forward(self, restore, sharp, blur):
        char = self.char(restore, sharp)
        edge = 0.05 * self.edge(restore, sharp)
        contrastive = 0.0005 * self.contrastive(restore, sharp, blur)
        loss = char + edge + contrastive
        return loss

class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(), k).unsqueeze(0).repeat(3, 1, 1, 1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.cuda()
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw // 2, kh // 2, kw // 2, kh // 2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered = self.conv_gauss(current)  # filter
        down = filtered[:, :, ::2, ::2]  # downsample
        new_filter = torch.zeros_like(filtered)
        new_filter[:, :, ::2, ::2] = down * 4  # upsample
        filtered = self.conv_gauss(new_filter)  # filter
        diff = current - filtered
        return diff

    def forward(self, x, y):
        # x = torch.clamp(x + 0.5, min = 0,max = 1)
        # y = torch.clamp(y + 0.5, min = 0,max = 1)
        loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss

class ContrastLoss(nn.Module):
    def __init__(self, ablation=False):

        super(ContrastLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.l1 = nn.L1Loss()
        self.ab = ablation
        self.down_sample_4 = nn.Upsample(scale_factor=1 / 4, mode='bilinear')
    def forward(self, restore, sharp, blur):
        B, C, H, W = restore.size()
        restore_vgg, sharp_vgg, blur_vgg = self.vgg(restore), self.vgg(sharp), self.vgg(blur)

        # filter out sharp regions
        threshold = 0.01
        mask = torch.mean(torch.abs(sharp-blur), dim=1).view(B, 1, H, W)
        mask[mask <= threshold] = 0
        mask[mask > threshold] = 1
        mask = self.down_sample_4(mask)
        d_ap = torch.mean(torch.abs((restore_vgg - sharp_vgg.detach())), dim=1).view(B, 1, H//4, W//4)
        d_an = torch.mean(torch.abs((restore_vgg - blur_vgg.detach())), dim=1).view(B, 1, H//4, W//4)
        mask_size = torch.sum(mask)
        contrastive = torch.sum((d_ap / (d_an + 1e-7)) * mask) / mask_size

        return contrastive

class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()

        for x in range(12):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        return h_relu1



