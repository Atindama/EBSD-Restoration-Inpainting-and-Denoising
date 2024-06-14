'''
This is the main function to implement the partial
convolution nerual network.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.common_types import _size_2_t
from torch.nn.modules.utils import _pair

# ==========THE MODEL==========


class PConv2d(nn.Conv2d):

    # Default hyperparameters are the same as those of nn.Conv2d.
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: _size_2_t, stride: _size_2_t = 1,
                 padding: _size_2_t | str = 0, dilation: _size_2_t = 1,
                 groups: int = 1, padding_mode: str = 'zeros') -> None:

        super(PConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                      stride, padding, dilation, groups,
                                      bias=True, padding_mode=padding_mode)

        kernel_size = _pair(kernel_size)  # int -> tuple[int, int] if necessary
        self.bias: Tensor

        # Define the mask update kernel. This is used to calculate M'. (See
        # equation (2) in the paper.)
        self.mask_update_kernel = torch.ones(out_channels,
                                             in_channels,
                                             *kernel_size)

        # Sum(1) for the kernel window.
        self.sum1 = in_channels*kernel_size[0]*kernel_size[1]

        # Initialize weights.
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, image: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:

        # Calculate new mask, equation (2) in paper.
        with torch.no_grad():

            # If anything within the kernel is 1, the convolution will be at
            # least 1 at the corresponing output point. Then just clamp the
            # output to be 0 or 1.
            new_mask = F.conv2d(mask, self.mask_update_kernel,
                                stride=self.stride, padding=self.padding,
                                dilation=self.dilation)
            # Also calculate sum(1)/sum(M), since newmask=sum(M) in each kernel position before the clamping
            sum_ratio = self.sum1 / (new_mask + 1e-8)
            new_mask = torch.clamp(new_mask, min=0, max=1)
            # sumratio is defined as 0 if sum(M)=0, even though the ratio itself blows up
            # So just set those to zero while keeping everything else constant with this multiplication
            sum_ratio = torch.mul(sum_ratio, new_mask)

        # Apply mask to image.
        masked_image = image * mask

        # Perform convolution.
        convolution_output = F.conv2d(masked_image, self.weight, self.bias, self.stride,
                                      self.padding, self.dilation, self.groups)

        # Perform renormalization and add bias. The bias is reshaped so it is
        # applied channel-wise and masked by new_mask.
        bias_reshaped = self.bias.view(1, self.out_channels, 1, 1)
        renormalized_image = (convolution_output - bias_reshaped) * sum_ratio
        output_image = renormalized_image + bias_reshaped
        output_image = output_image * new_mask

        return output_image, new_mask


class UpCat(nn.Module):
    def __init__(self, scale: int = 2) -> None:
        super(UpCat, self).__init__()

        self.upsample = nn.Upsample(scale_factor=scale, mode='nearest')

    def forward(self, encoder_el: Tensor, decoder_el: Tensor,
                encoder_mask: Tensor, decoder_mask: Tensor
                ) -> tuple[Tensor, Tensor]:
        new_image = self.upsample(decoder_el)
        new_image = torch.cat([new_image, encoder_el], dim=1)

        new_mask = self.upsample(decoder_mask)
        new_mask = torch.cat([new_mask, encoder_mask], dim=1)

        return new_image, new_mask


class EncoderLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: _size_2_t = (3, 3),
                 padding: _size_2_t | str = 1, bnorm: bool = True) -> None:
        super(EncoderLayer, self).__init__()

        self.pconv = PConv2d(in_channels, out_channels,
                             kernel_size=kernel_size, stride=2,
                             padding=padding)
        self.bnorm = nn.BatchNorm2d(out_channels) if bnorm else nn.Identity()
        self.activ = nn.ReLU()

    def forward(self, image: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        output_image, output_mask = self.pconv(image, mask)
        output_image = self.bnorm(self.activ(output_image))
        return output_image, output_mask


class DecoderLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: _size_2_t = (3, 3),
                 bnorm: bool = True,
                 activ: nn.Module = nn.LeakyReLU(negative_slope=0.2)) -> None:
        super(DecoderLayer, self).__init__()

        self.pconv = PConv2d(in_channels, out_channels, kernel_size,
                             padding='same')
        self.bnorm = nn.BatchNorm2d(out_channels) if bnorm else nn.Identity()
        self.activ = activ
        self.upcat = UpCat(scale=2)

    def forward(self, left_im: Tensor, lower_im: Tensor, left_mask: Tensor,
                lower_mask: Tensor) -> tuple[Tensor, Tensor]:
        output_image, output_mask = self.upcat(
            left_im, lower_im, left_mask, lower_mask)
        output_image, output_mask = self.pconv(output_image, output_mask)
        output_image = self.bnorm(self.activ(output_image))
        return output_image, output_mask


class Model(nn.Module):
    def __init__(self) -> None:
        super(Model, self).__init__()

        self.enc_1 = EncoderLayer(3, 64, kernel_size=7, padding=3, bnorm=False)
        self.enc_2 = EncoderLayer(64, 128, kernel_size=5, padding=2)
        self.enc_3 = EncoderLayer(128, 256, kernel_size=5, padding=2)
        self.enc_4 = EncoderLayer(256, 512)
        self.enc_5 = EncoderLayer(512, 512)
        self.enc_6 = EncoderLayer(512, 512)
        self.enc_7 = EncoderLayer(512, 512)

        self.dec_1 = DecoderLayer(512+512, 512)
        self.dec_2 = DecoderLayer(512+512, 512)
        self.dec_3 = DecoderLayer(512+512, 512)
        self.dec_4 = DecoderLayer(512+256, 256)
        self.dec_5 = DecoderLayer(256+128, 128)
        self.dec_6 = DecoderLayer(128+64, 64)
        self.dec_7 = DecoderLayer(64+3, 3, activ=nn.Identity(), bnorm=False)

    def forward(self, image, mask):
        enc_img1, enc_mask1 = self.enc_1(image, mask)
        enc_img2, enc_mask2 = self.enc_2(enc_img1, enc_mask1)
        enc_img3, enc_mask3 = self.enc_3(enc_img2, enc_mask2)
        enc_img4, enc_mask4 = self.enc_4(enc_img3, enc_mask3)
        enc_img5, enc_mask5 = self.enc_5(enc_img4, enc_mask4)
        enc_img6, enc_mask6 = self.enc_6(enc_img5, enc_mask5)
        enc_img7, enc_mask7 = self.enc_7(enc_img6, enc_mask6)

        dec_img, dec_mask = self.dec_1(
            enc_img6, enc_img7, enc_mask6, enc_mask7)
        dec_img, dec_mask = self.dec_2(enc_img5, dec_img, enc_mask5, dec_mask)
        dec_img, dec_mask = self.dec_3(enc_img4, dec_img, enc_mask4, dec_mask)
        dec_img, dec_mask = self.dec_4(enc_img3, dec_img, enc_mask3, dec_mask)
        dec_img, dec_mask = self.dec_5(enc_img2, dec_img, enc_mask2, dec_mask)
        dec_img, dec_mask = self.dec_6(enc_img1, dec_img, enc_mask1, dec_mask)
        dec_img, dec_mask = self.dec_7(image, dec_img, mask, dec_mask)

        return dec_img, dec_mask
