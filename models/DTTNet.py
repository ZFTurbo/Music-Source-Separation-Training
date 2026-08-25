import torch
import torch.nn as nn
from functools import partial
from utils.model_utils import prefer_target_instrument


class STFT:
    def __init__(self, config):
        self.n_fft = config.n_fft
        self.hop_length = config.hop_length
        self.window = torch.hann_window(window_length=self.n_fft, periodic=True)
        self.dim_f = config.dim_f
        self.length = config.chunk_size
        self.channels = config.num_channels

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
            center=True,
            length=self.length
        )
        x = x.reshape([*batch_dims, self.channels, -1])

        
        return x

def get_norm(norm_type):
    def norm(c, norm_type):
        if norm_type == 'BatchNorm' or norm_type == 'BN':
            return nn.BatchNorm2d(c)
        elif norm_type == 'InstanceNorm':
            return nn.InstanceNorm2d(c, affine=True)
        elif 'GroupNorm' in norm_type:
            g = int(norm_type.replace('GroupNorm', ''))
            return nn.GroupNorm(num_groups=g, num_channels=c)
        else:
            return nn.Identity()

    return partial(norm, norm_type=norm_type)

def get_act(act_type):
    if act_type == 'gelu':
        return nn.GELU()
    elif act_type == 'relu':
        return nn.ReLU(inplace=True)
    elif act_type in ('swish', 'silu'):
        return nn.SiLU()
    elif act_type in ('hardswish'):
        return nn.Hardswish()
    elif act_type == 'prelu':
        return nn.PReLU()
    elif act_type[:3] == 'elu':
        alpha = float(act_type.replace('elu', ''))
        return nn.ELU(alpha, inplace=True)
    else:
        raise Exception(f"Unknown activation type: {act_type}")

class RNNModule(nn.Module):
    """
    RNN submodule of BandSequence module
    """

    def __init__(
            self,
            group_num: int,
            input_dim_size: int,
            hidden_dim_size: int,
            rnn_type: str = 'lstm',
            bidirectional: bool = True
    ):
        super(RNNModule, self).__init__()
        self.groupnorm = nn.GroupNorm(group_num, input_dim_size)
        self.rnn = getattr(nn, rnn_type)(
            input_dim_size, hidden_dim_size, batch_first=True, bidirectional=bidirectional # 输出是2*hidden_dim_size，因为是bi
        )
        self.fc = nn.Linear(
            hidden_dim_size * 2 if bidirectional else hidden_dim_size,
            input_dim_size
        )

    def forward(
            self,
            x: torch.Tensor
    ):
        """
        Input shape:
            across T - [batch_size, k_subbands, time, n_features]
            OR
            across K - [batch_size, time, k_subbands, n_features]
        """
        B, K, T, N = x.shape  # across T      across K (keep in mind T->K, K->T)
        # print(x.shape)

        out = x.view(B * K, T, N)  # [BK, T, N]    [BT, K, N]

        # print(out.shape)
        # print(self.groupnorm)
        out = self.groupnorm(
            out.transpose(-1, -2)
        ).transpose(-1, -2)  # [BK, T, N]    [BT, K, N]
        out = self.rnn(out)[0]  # [BK, T, H]    [BT, K, H]， 最后一维是特征
        out = self.fc(out)  # [BK, T, N]    [BT, K, N]

        x = out.view(B, K, T, N) + x  # [B, K, T, N]  [B, T, K, N]

        x = x.permute(0, 2, 1, 3).contiguous()  # [B, T, K, N]  [B, K, T, N]
        return x

class BandSequenceModelModule(nn.Module):
    """
    BandSequence (2nd) Module of BandSplitRNN.
    Runs input through n BiLSTMs in two dimensions - time and subbands.
    """

    def __init__(
            self,
            # group_num,
            input_dim_size: int,
            hidden_dim_size: int,
            rnn_type: str = 'lstm',
            bidirectional: bool = True,
            num_layers: int = 12,
            n_heads: int = 4,
    ):
        super(BandSequenceModelModule, self).__init__()

        self.bsrnn = nn.ModuleList([])
        self.n_heads = n_heads

        input_dim_size = input_dim_size // n_heads
        hidden_dim_size = hidden_dim_size // n_heads
        group_num = input_dim_size // 16
        # print(f"input_dim_size: {input_dim_size}, hidden_dim_size: {hidden_dim_size}, group_num: {group_num}")

        # print(group_num, input_dim_size)

        for _ in range(num_layers):
            rnn_across_t = RNNModule(
                group_num, input_dim_size, hidden_dim_size, rnn_type, bidirectional
            )
            rnn_across_k = RNNModule(
                group_num, input_dim_size, hidden_dim_size, rnn_type, bidirectional
            )
            self.bsrnn.append(
                nn.Sequential(rnn_across_t, rnn_across_k)
            )

    def forward(self, x: torch.Tensor):
        """
        Input shape: [batch_size, k_subbands, time, n_features]
        Output shape: [batch_size, k_subbands, time, n_features]
        """
        # x (b,c,t,f)
        b,c,t,f = x.shape
        x = x.view(b * self.n_heads, c // self.n_heads, t, f) # [b*n_heads, c//n_heads, t, f]

        x = x.permute(0, 3, 2, 1).contiguous()  # [b*n_heads, f, t, c//n_heads]
        for i in range(len(self.bsrnn)):
            x = self.bsrnn[i](x)

        x = x.permute(0, 3, 2, 1).contiguous()  # [b*n_heads, c//n_heads, t, f]
        x = x.view(b, c, t, f)  # [b, c, t, f] 
        return x

class TFC(nn.Module):
    def __init__(self, c_in, c_out, l, k, norm, act):
        super(TFC, self).__init__()

        self.H = nn.ModuleList()
        for i in range(l):
            self.H.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=c_out if i > 0 else c_in, out_channels=c_out, kernel_size=k, stride=1, padding=k // 2),
                    norm(c_out if i > 0 else c_in),
                    act,
                )
            )

    def forward(self, x):
        for h in self.H:
            x = h(x)
        return x

class DenseTFC(nn.Module):
    def __init__(self, c_in, c_out, l, k, norm, act):
        super(DenseTFC, self).__init__()

        self.conv = nn.ModuleList()
        for i in range(l):
            self.conv.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=k, stride=1, padding=k // 2),
                    norm(c_out),
                    act,
                )
            )

    def forward(self, x):
        for layer in self.conv[:-1]:
            x = torch.cat([layer(x), x], 1)
        return self.conv[-1](x)

class TFC_TDF(nn.Module):
    def __init__(self, c_in, c_out, l, f, k, bn, norm, act, dense=False, bias=True):
        super(TFC_TDF, self).__init__()

        self.use_tdf = bn is not None
        self.tfc = DenseTFC(c_in, c_out, l, k, norm, act) if dense else TFC(c_in, c_out, l, k, norm, act)

        if self.use_tdf:
            if bn == 0:
                self.tdf = nn.Sequential(
                    nn.Linear(f, f, bias=bias),
                    norm(c_out),
                    act,
                )
            else:
                self.tdf = nn.Sequential(
                    nn.Linear(f, f // bn, bias=bias),
                    norm(c_out),
                    act,
                    nn.Linear(f // bn, f, bias=bias),
                    norm(c_out),
                    act,
                    
                )

    def forward(self, x):
        x = self.tfc(x)
        return x + self.tdf(x) if self.use_tdf else x

class TFC_TDF_Res1(nn.Module):
    def __init__(self, c_in, c_out, l, f, k, bn, norm, act, dense=False, bias=True):
        super(TFC_TDF_Res1, self).__init__()

        self.use_tdf = bn is not None

        self.tfc = DenseTFC(c_in, c_out, l, k, norm, act) if dense else TFC(c_in, c_out, l, k, norm, act)

        self.res = TFC(c_in, c_out, 1, k, norm, act)

        if self.use_tdf:
            if bn == 0:
                self.tdf = nn.Sequential(
                    nn.Linear(f, f, bias=bias),
                    norm(c_out),
                    act
                )
            else:
                self.tdf = nn.Sequential(
                    nn.Linear(f, f // bn, bias=bias),
                    norm(c_out),
                    act,
                    nn.Linear(f // bn, f, bias=bias),
                    norm(c_out),
                    act
                )

    def forward(self, x):
        res = self.res(x)
        x = self.tfc(x)
        x = x + res
        return x + self.tdf(x) if self.use_tdf else x

class TFC_TDF_Res2(nn.Module):
    def __init__(self, c_in, c_out, l, f, k, bn, norm, act, dense=False, bias=True):
        super(TFC_TDF_Res2, self).__init__()

        self.use_tdf = bn is not None

        self.tfc1 = TFC(c_in, c_out, l, k, norm, act)
        self.tfc2 = TFC(c_in, c_out, l, k, norm, act)

        self.res = TFC(c_in, c_out, 1, k, norm, act)

        if self.use_tdf:
            if bn == 0:
                self.tdf = nn.Sequential(
                    nn.Linear(f, f, bias=bias),
                    norm(c_out),
                    act
                )
            else:
                self.tdf = nn.Sequential(
                    nn.Linear(f, f // bn, bias=bias),
                    norm(c_out),
                    act,
                    nn.Linear(f // bn, f, bias=bias),
                    norm(c_out),
                    act,
                )

    def forward(self, x):
        res = self.res(x)
        x = self.tfc1(x)
        if self.use_tdf:
            x = x + self.tdf(x)
        x = self.tfc2(x)
        x = x + res
        return x

# ----------------------------
# New: TFC-TDF v3 block
# ----------------------------
class TFC_TDF_v3(nn.Module):
    """
    A v3-style residual TFC-TDF block:
      - PreNorm + Act before each conv (pre-activation style)
      - Two Conv2d layers per residual unit
      - Optional TDF (frequency MLP) injected between convs
      - 1x1 shortcut when c_in != c_out
      - Repeated `l` times as stacked residual units
    Shape convention inside the network is (B, C, T, F) so Linear(...) operates on the last dim (F).
    """
    def __init__(self, c_in, c_out, l, f, k, bn, norm, act, dense=False, bias=True):
        super().__init__()
        self.units = nn.ModuleList()
        use_tdf = bn is not None

        def make_tdf(channels: int):
            if not use_tdf:
                return None
            if bn == 0:
                return nn.Sequential(
                    nn.Linear(f, f, bias=bias),
                    norm(channels),
                    act,
                )
            else:
                return nn.Sequential(
                    nn.Linear(f, f // bn, bias=bias),
                    norm(channels),
                    act,
                    nn.Linear(f // bn, f, bias=bias),
                    norm(channels),
                    act,
                )

        Cin = c_in
        for _ in range(l):
            unit = nn.ModuleDict({
                "pre1": nn.Sequential(norm(Cin), act),
                "conv1": nn.Conv2d(Cin, c_out, k, stride=1, padding=k // 2, bias=False),
                "pre2": nn.Sequential(norm(c_out), act),
                "tdf": make_tdf(c_out),
                "conv2": nn.Conv2d(c_out, c_out, k, stride=1, padding=k // 2, bias=False),
                "shortcut": nn.Identity() if Cin == c_out else nn.Conv2d(Cin, c_out, 1, bias=False),
            })
            self.units.append(unit)
            Cin = c_out  # subsequent units are c_out → c_out

    def forward(self, x):
        for u in self.units:
            res = u["shortcut"](x)
            y = u["pre1"](x)
            y = u["conv1"](y)
            y = u["pre2"](y)
            if u["tdf"] is not None:
                # TDF acts along frequency dimension (last dim) at each time step
                y = y + u["tdf"](y)
            y = u["conv2"](y)
            x = y + res
        return x


class DPTDFNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        norm = get_norm(norm_type=getattr(config.model, 'norm', 'BN')) #original paper uses batch norm
        act = get_act(act_type=config.model.act)

        self.num_target_instruments = len(prefer_target_instrument(config))

        # Get model parameters from config
        self.num_blocks = config.model.num_blocks
        self.l = config.model.l
        self.g = config.model.g
        self.k = config.model.k
        self.bn = config.model.bn
        self.bias = config.model.bias
        self.block_type = config.model.block_type

        # Audio parameters
        dim_c = config.audio.num_channels * 2
        f = config.audio.dim_f

        self.n = self.num_blocks // 2
        scale = (2, 2)

        # Select block type
        if self.block_type == "TFC_TDF":
            T_BLOCK = TFC_TDF
        elif self.block_type == "TFC_TDF_Res1":
            T_BLOCK = TFC_TDF_Res1
        elif self.block_type == "TFC_TDF_Res2":
            T_BLOCK = TFC_TDF_Res2
        elif self.block_type == "TFC_TDF_v3":
            T_BLOCK = TFC_TDF_v3
        else:
            raise ValueError(f"Unknown block type {self.block_type}")

        self.first_conv = nn.Sequential(
            nn.Conv2d(in_channels=dim_c, out_channels=self.g, kernel_size=(1, 1)),
            norm(self.g),
            act,
        )

        c = self.g
        self.encoding_blocks = nn.ModuleList()
        self.ds = nn.ModuleList()

        for i in range(self.n):
            c_in = c
            self.encoding_blocks.append(T_BLOCK(c_in, c, self.l, f, self.k, self.bn, norm, act, bias=self.bias))
            self.ds.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=c, out_channels=c + self.g, kernel_size=scale, stride=scale),
                    norm(c + self.g),
                    act,
                )
            )
            f = f // 2
            c += self.g

        self.bottleneck_block1 = T_BLOCK(c, c, self.l, f, self.k, self.bn, norm, act, bias=self.bias)
        
        bs_config = config.model.bandsequence
        self.bottleneck_block2 = BandSequenceModelModule(
            input_dim_size=c,
            hidden_dim_size=2 * c,
            rnn_type=bs_config.rnn_type,
            bidirectional=bs_config.bidirectional,
            num_layers=bs_config.num_layers,
            n_heads=bs_config.n_heads,
        )


        self.decoding_blocks = nn.ModuleList()
        self.us = nn.ModuleList()
        for i in range(self.n):
            # print(f"i: {i}, in channels: {c}")
            self.us.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels=c, out_channels=c - self.g, kernel_size=scale, stride=scale),
                    norm(c - self.g),
                    act,
                )
            )

            f = f * 2
            c -= self.g

            self.decoding_blocks.append(T_BLOCK(c, c, self.l, f, self.k, self.bn, norm, act, bias=self.bias))

        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=self.num_target_instruments * dim_c, kernel_size=(1, 1)),
        )

        self.stft = STFT(config.audio)

    def forward(self, x):
        """
        Args:
            x: (batch, channels, time)
        """

        # print(f"x.shape0 : {x.shape}\n")
        x = self.stft(x)
        # print(f"x.shape stft : {x.shape}\n")

        first_conv_out = x = self.first_conv(x)
        # print(f"x.shape1 : {x.shape}\n")
        
        x = x.transpose(-1, -2)
        # print(f"x.shape2 : {x.shape}\n")     
        
        ds_outputs = []
        for i in range(self.n):
            x = self.encoding_blocks[i](x)
            ds_outputs.append(x)
            x = self.ds[i](x)

        # print(f"bottleneck in: {x.shape}")
        x = self.bottleneck_block1(x)
        x = self.bottleneck_block2(x)

        for i in range(self.n):
            x = self.us[i](x)
            # print(f"us{i} in: {x.shape}")
            # print(f"ds{i} out: {ds_outputs[-i - 1].shape}")
            x = x * ds_outputs[-i - 1]
            x = self.decoding_blocks[i](x)

        x = x.transpose(-1, -2)
        # print(f"x.shape3 : {x.shape}\n")


        # x = x * first_conv_out  # reduce artifacts mdx23c style (added by myself but not tried yet, so disabled for now)
         
        x = self.final_conv(x)
        # print(f"x.shape4 : {x.shape}\n")
        
        
        # reshaping with added num_instruments dim
        b, c, f, t = x.shape
        x = x.reshape(b, self.num_target_instruments, -1, f, t)
        # print(f"x.shape5 : {x.shape}\n")
        
        x = self.stft.inverse(x)
        # print(f"x.shape6 : {x.shape}\n")
        
        return x
