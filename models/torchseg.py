import torch
import torch.nn as nn
import torchseg as smp


class STFT:
    def __init__(self, config):
        self.n_fft = config.n_fft
        self.hop_length = config.hop_length
        self.window = torch.hann_window(window_length=self.n_fft, periodic=True)
        self.dim_f = config.dim_f

    def __call__(self, x):
        window = self.window.to(x.device)
        batch_dims = x.shape[:-2]
        c, t = x.shape[-2:]
        x = x.reshape([-1, t])
        x = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            center=True,
            return_complex=True
        )
        x = torch.view_as_real(x)
        x = x.permute([0, 3, 1, 2])
        x = x.reshape([*batch_dims, c, 2, -1, x.shape[-1]]).reshape([*batch_dims, c * 2, -1, x.shape[-1]])
        return x[..., :self.dim_f, :]

    def inverse(self, x):
        window = self.window.to(x.device)
        batch_dims = x.shape[:-3]
        c, f, t = x.shape[-3:]
        n = self.n_fft // 2 + 1
        f_pad = torch.zeros([*batch_dims, c, n - f, t]).to(x.device)
        x = torch.cat([x, f_pad], -2)
        x = x.reshape([*batch_dims, c // 2, 2, n, t]).reshape([-1, 2, n, t])
        x = x.permute([0, 2, 3, 1])
        x = x[..., 0] + x[..., 1] * 1.j
        x = torch.istft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            center=True
        )
        x = x.reshape([*batch_dims, 2, -1])
        return x


def get_act(act_type):
    if act_type == 'gelu':
        return nn.GELU()
    elif act_type == 'relu':
        return nn.ReLU()
    elif act_type[:3] == 'elu':
        alpha = float(act_type.replace('elu', ''))
        return nn.ELU(alpha)
    else:
        raise Exception


def get_decoder(config, c):
    decoder = None
    decoder_options = dict()
    if config.model.decoder_type == 'unet':
        try:
            decoder_options = dict(config.decoder_unet)
        except:
            pass
        decoder = smp.Unet(
            encoder_name=config.model.encoder_name,
            encoder_weights="imagenet",
            in_channels=c,
            classes=c,
            **decoder_options,
        )
    elif config.model.decoder_type == 'fpn':
        try:
            decoder_options = dict(config.decoder_fpn)
        except:
            pass
        decoder = smp.FPN(
            encoder_name=config.model.encoder_name,
            encoder_weights="imagenet",
            in_channels=c,
            classes=c,
            **decoder_options,
        )
    elif config.model.decoder_type == 'unet++':
        try:
            decoder_options = dict(config.decoder_unet_plus_plus)
        except:
            pass
        decoder = smp.UnetPlusPlus(
            encoder_name=config.model.encoder_name,
            encoder_weights="imagenet",
            in_channels=c,
            classes=c,
            **decoder_options,
        )
    elif config.model.decoder_type == 'manet':
        try:
            decoder_options = dict(config.decoder_manet)
        except:
            pass
        decoder = smp.MAnet(
            encoder_name=config.model.encoder_name,
            encoder_weights="imagenet",
            in_channels=c,
            classes=c,
            **decoder_options,
        )
    elif config.model.decoder_type == 'linknet':
        try:
            decoder_options = dict(config.decoder_linknet)
        except:
            pass
        decoder = smp.Linknet(
            encoder_name=config.model.encoder_name,
            encoder_weights="imagenet",
            in_channels=c,
            classes=c,
            **decoder_options,
        )
    elif config.model.decoder_type == 'pspnet':
        try:
            decoder_options = dict(config.decoder_pspnet)
        except:
            pass
        decoder = smp.PSPNet(
            encoder_name=config.model.encoder_name,
            encoder_weights="imagenet",
            in_channels=c,
            classes=c,
            **decoder_options,
        )
    elif config.model.decoder_type == 'pspnet':
        try:
            decoder_options = dict(config.decoder_pspnet)
        except:
            pass
        decoder = smp.PSPNet(
            encoder_name=config.model.encoder_name,
            encoder_weights="imagenet",
            in_channels=c,
            classes=c,
            **decoder_options,
        )
    elif config.model.decoder_type == 'pan':
        try:
            decoder_options = dict(config.decoder_pan)
        except:
            pass
        decoder = smp.PAN(
            encoder_name=config.model.encoder_name,
            encoder_weights="imagenet",
            in_channels=c,
            classes=c,
            **decoder_options,
        )
    elif config.model.decoder_type == 'deeplabv3':
        try:
            decoder_options = dict(config.decoder_deeplabv3)
        except:
            pass
        decoder = smp.DeepLabV3(
            encoder_name=config.model.encoder_name,
            encoder_weights="imagenet",
            in_channels=c,
            classes=c,
            **decoder_options,
        )
    elif config.model.decoder_type == 'deeplabv3plus':
        try:
            decoder_options = dict(config.decoder_deeplabv3plus)
        except:
            pass
        decoder = smp.DeepLabV3Plus(
            encoder_name=config.model.encoder_name,
            encoder_weights="imagenet",
            in_channels=c,
            classes=c,
            **decoder_options,
        )
    return decoder


class Torchseg_Net(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        act = get_act(act_type=config.model.act)

        self.num_target_instruments = 1 if config.training.target_instrument else len(config.training.instruments)
        self.num_subbands = config.model.num_subbands

        dim_c = self.num_subbands * config.audio.num_channels * 2
        c = config.model.num_channels
        f = config.audio.dim_f // self.num_subbands

        self.first_conv = nn.Conv2d(dim_c, c, 1, 1, 0, bias=False)

        self.unet_model = get_decoder(config, c)

        self.final_conv = nn.Sequential(
            nn.Conv2d(c + dim_c, c, 1, 1, 0, bias=False),
            act,
            nn.Conv2d(c, self.num_target_instruments * dim_c, 1, 1, 0, bias=False)
        )

        self.stft = STFT(config.audio)

    def cac2cws(self, x):
        k = self.num_subbands
        b, c, f, t = x.shape
        x = x.reshape(b, c, k, f // k, t)
        x = x.reshape(b, c * k, f // k, t)
        return x

    def cws2cac(self, x):
        k = self.num_subbands
        b, c, f, t = x.shape
        x = x.reshape(b, c // k, k, f, t)
        x = x.reshape(b, c // k, f * k, t)
        return x

    def forward(self, x):

        x = self.stft(x)

        mix = x = self.cac2cws(x)

        first_conv_out = x = self.first_conv(x)

        x = x.transpose(-1, -2)

        x = self.unet_model(x)

        x = x.transpose(-1, -2)

        x = x * first_conv_out  # reduce artifacts

        x = self.final_conv(torch.cat([mix, x], 1))

        x = self.cws2cac(x)

        if self.num_target_instruments > 1:
            b, c, f, t = x.shape
            x = x.reshape(b, self.num_target_instruments, -1, f, t)

        x = self.stft.inverse(x)
        return x
