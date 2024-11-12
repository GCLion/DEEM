import random
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class CONV(nn.Module):
    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10**(mel / 2595) - 1)

    def __init__(self,
                 out_channels,
                 kernel_size,
                 sample_rate=16000,
                 in_channels=1,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=False,
                 groups=1,
                 mask=False):
        super().__init__()
        if in_channels != 1:

            msg = "SincConv only support one input channel (here, in_channels = {%i})" % (
                in_channels)
            raise ValueError(msg)
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate

        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        if kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.mask = mask
        if bias:
            raise ValueError('SincConv does not support bias.')
        if groups > 1:
            raise ValueError('SincConv does not support groups.')

        NFFT = 512
        f = int(self.sample_rate / 2) * np.linspace(0, 1, int(NFFT / 2) + 1)
        fmel = self.to_mel(f)
        fmelmax = np.max(fmel)
        fmelmin = np.min(fmel)
        filbandwidthsmel = np.linspace(fmelmin, fmelmax, self.out_channels + 1)
        filbandwidthsf = self.to_hz(filbandwidthsmel)

        self.mel = filbandwidthsf
        self.hsupp = torch.arange(-(self.kernel_size - 1) / 2,
                                  (self.kernel_size - 1) / 2 + 1)
        self.band_pass = torch.zeros(self.out_channels, self.kernel_size)
        for i in range(len(self.mel) - 1):
            fmin = self.mel[i]
            fmax = self.mel[i + 1]
            hHigh = (2*fmax/self.sample_rate) * \
                np.sinc(2*fmax*self.hsupp/self.sample_rate)
            hLow = (2*fmin/self.sample_rate) * \
                np.sinc(2*fmin*self.hsupp/self.sample_rate)
            hideal = hHigh - hLow

            self.band_pass[i, :] = Tensor(np.hamming(
                self.kernel_size)) * Tensor(hideal)

    def forward(self, x, mask=False):
        band_pass_filter = self.band_pass.clone().to(x.device)
        if mask:
            A = np.random.uniform(0, 20)
            A = int(A)
            A0 = random.randint(0, band_pass_filter.shape[0] - A)
            band_pass_filter[A0:A0 + A, :] = 0
        else:
            band_pass_filter = band_pass_filter

        self.filters = (band_pass_filter).view(self.out_channels, 1,
                                               self.kernel_size)

        return F.conv1d(x,
                        self.filters,
                        stride=self.stride,
                        padding=self.padding,
                        dilation=self.dilation,
                        bias=None,
                        groups=1)


class Residual_block(nn.Module):
    def __init__(self, nb_filts, first=False):
        super().__init__()
        self.first = first

        if not self.first:
            self.bn1 = nn.BatchNorm2d(num_features=nb_filts[0])
        self.conv1 = nn.Conv2d(in_channels=nb_filts[0],
                               out_channels=nb_filts[1],
                               kernel_size=(2, 3),
                               padding=(1, 1),
                               stride=1)
        self.selu = nn.SELU(inplace=True)

        self.bn2 = nn.BatchNorm2d(num_features=nb_filts[1])
        self.conv2 = nn.Conv2d(in_channels=nb_filts[1],
                               out_channels=nb_filts[1],
                               kernel_size=(2, 3),
                               padding=(0, 1),
                               stride=1)

        if nb_filts[0] != nb_filts[1]:
            self.downsample = True
            self.conv_downsample = nn.Conv2d(in_channels=nb_filts[0],
                                             out_channels=nb_filts[1],
                                             padding=(0, 1),
                                             kernel_size=(1, 3),
                                             stride=1)

        else:
            self.downsample = False
        self.mp = nn.MaxPool2d((1, 3))  # self.mp = nn.MaxPool2d((1,4))

    def forward(self, x):
        identity = x
        if not self.first:
            out = self.bn1(x)
            out = self.selu(out)
        else:
            out = x
        out = self.conv1(x)

        out = self.bn2(out)
        out = self.selu(out)
        out = self.conv2(out)
        if self.downsample:
            identity = self.conv_downsample(identity)

        out += identity
        out = self.mp(out)
        return out




class Raw2feat(nn.Module):
    def __init__(self, basemodel):
        super(Raw2feat, self).__init__()
        '''
        Sinc conv. layer
        '''
        self.conv_time = basemodel.conv_time

        self.first_bn = basemodel.first_bn

        self.selu = basemodel.selu
        del basemodel
        #
    def forward(self, x,Freq_aug=False):
        """
        x= (#bs,samples)
        """

        # nb_samp = x.shape[0]
        # len_seq = x.shape[1]
        #
        # x = x.view(nb_samp, 1, len_seq)


        x = x.unsqueeze(1)
        x = self.conv_time(x, mask=Freq_aug)
        x = x.unsqueeze(dim=1)
        x = F.max_pool2d(torch.abs(x), (3, 3))
        x = self.first_bn(x)
        x = self.selu(x)
        return x


def spec_deconv2d():
    deconv_layers = []
    deconv_layers+=[nn.ConvTranspose2d(in_channels=64,out_channels=32,kernel_size=(1,3),stride=(1,2)),nn.BatchNorm2d(32),nn.Tanh()]
    deconv_layers+=[nn.ConvTranspose2d(in_channels=32,out_channels=1,kernel_size=(1,3),stride=(1,2)),nn.Tanh()]
    return nn.Sequential(*deconv_layers)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        # 设定网络
        self.latent_to_hidden = nn.Linear(58, 1024)
        self.decoder = spec_deconv2d()
        self.hidden_to_output = nn.Linear(4099, 21490)
        self.ReLU = nn.ReLU()

    def forward(self, latent):
        hidden = self.ReLU(self.latent_to_hidden(latent))
        hidden = self.decoder(hidden)
        output = self.hidden_to_output(hidden)

        return output


class Autoencoder_decoupling(nn.Module):
    def __init__(self, model,weight,d_args):
        super(Autoencoder_decoupling, self).__init__()
        """
          nb_samp: 64600
          first_conv: 1024   # no. of filter coefficients
          in_channels: 1
          filts: [20, [20, 20], [20, 128], [128, 128]] # no. of filters channel in residual blocks
        """
        self.d_args = d_args
        filts = d_args["filts"]
        # gat_dims = d_args["gat_dims"]
        # pool_ratios = d_args["pool_ratios"]
        # temperatures = d_args["temperatures"]

        self.Raw2feat = Raw2feat(model)
        del model
        self.encoder1 = nn.Sequential(
            nn.Sequential(Residual_block(nb_filts=filts[1], first=True)),
            nn.Sequential(Residual_block(nb_filts=filts[2])),
            nn.Sequential(Residual_block(nb_filts=filts[3])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])))
        self.encoder2 = nn.Sequential(
            nn.Sequential(Residual_block(nb_filts=filts[1], first=True)),
            nn.Sequential(Residual_block(nb_filts=filts[2])),
            nn.Sequential(Residual_block(nb_filts=filts[3])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])))

        self.encoder1.load_state_dict(weight)
        self.encoder2.load_state_dict(weight)
        self.decoder3 = Decoder()

    def forward(self, x1, x2, y1, y2):
        x1 = self.Raw2feat(x1)
        x2 = self.Raw2feat(x2)
        y1 = self.Raw2feat(y1)
        y2 = self.Raw2feat(y2)
        latentx11 = self.encoder1(x1)
        latentx12 = self.encoder2(x1)
        latentx1 = torch.cat((latentx11, latentx12), -1)
        recon_x1 = self.decoder3(latentx1)

        latentx21 = self.encoder1(x2)
        latentx22 = self.encoder2(x2)
        latentx2 = torch.cat((latentx21, latentx22), -1)
        recon_x2 = self.decoder3(latentx2)

        latentx1_ = torch.cat((latentx11, latentx22), -1)
        latentx2_ = torch.cat((latentx21, latentx12), -1)
        recon_x1_ = self.decoder3(latentx1_)
        recon_x2_ = self.decoder3(latentx2_)

        latenty11 = self.encoder1(y1)
        latenty12 = self.encoder2(y1)
        latenty1 = torch.cat((latenty11, latenty12), -1)
        recon_y1 = self.decoder3(latenty1)

        latenty21 = self.encoder1(y2)
        latenty22 = self.encoder2(y2)
        latenty2 = torch.cat((latenty21, latenty22), -1)
        recon_y2 = self.decoder3(latenty2)

        latenty1_ = torch.cat((latenty11, latenty22), -1)
        latenty2_ = torch.cat((latenty21, latenty12), -1)
        recon_y1_ = self.decoder3(latenty1_)
        recon_y2_ = self.decoder3(latenty2_)

        latentry11 = self.encoder1(recon_y1_)
        latentry12 = self.encoder2(recon_y1_)
        latentry21 = self.encoder1(recon_y2_)
        latentry22 = self.encoder2(recon_y2_)
        latentry1 = torch.cat((latentry11, latentry22), -1)
        latentry2 = torch.cat((latentry21, latentry12), -1)
        recon_y1__ = self.decoder3(latentry1)
        recon_y2__ = self.decoder3(latentry2)

        return x1, x2, y1, y2, recon_x1, recon_x2, recon_x1_, recon_x2_, recon_y1, recon_y2, recon_y1__, recon_y2__
