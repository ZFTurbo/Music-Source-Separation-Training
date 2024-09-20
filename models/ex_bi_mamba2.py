# https://github.com/Human9000/nd-Mamba2-torch

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from abc import abstractmethod


def silu(x):
    return x * F.sigmoid(x)


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x, z):
        x = x * silu(z)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class Mamba2(nn.Module):
    def __init__(self, d_model: int,  # model dimension (D)
                 n_layer: int = 24,  # number of Mamba-2 layers in the language model
                 d_state: int = 128,  # state dimension (N)
                 d_conv: int = 4,  # convolution kernel size
                 expand: int = 2,  # expansion factor (E)
                 headdim: int = 64,  # head dimension (P)
                 chunk_size: int = 64,  # matrix partition size (Q)
                 ):
        super().__init__()
        self.n_layer = n_layer
        self.d_state = d_state
        self.headdim = headdim
        # self.chunk_size = torch.tensor(chunk_size, dtype=torch.int32)
        self.chunk_size = chunk_size

        self.d_inner = expand * d_model
        assert self.d_inner % self.headdim == 0, "self.d_inner must be divisible by self.headdim"
        self.nheads = self.d_inner // self.headdim

        d_in_proj = 2 * self.d_inner + 2 * self.d_state + self.nheads
        self.in_proj = nn.Linear(d_model, d_in_proj, bias=False)

        conv_dim = self.d_inner + 2 * d_state
        self.conv1d = nn.Conv1d(conv_dim, conv_dim, d_conv, groups=conv_dim, padding=d_conv - 1, )
        self.dt_bias = nn.Parameter(torch.empty(self.nheads, ))
        self.A_log = nn.Parameter(torch.empty(self.nheads, ))
        self.D = nn.Parameter(torch.empty(self.nheads, ))
        self.norm = RMSNorm(self.d_inner, )
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False, )

    def forward(self, u: Tensor):
        A = -torch.exp(self.A_log)  # (nheads,)
        zxbcdt = self.in_proj(u)  # (batch, seqlen, d_in_proj)
        z, xBC, dt = torch.split(
            zxbcdt,
            [
                self.d_inner,
                self.d_inner + 2 * self.d_state,
                self.nheads,
            ],
            dim=-1,
        )
        dt = F.softplus(dt + self.dt_bias)  # (batch, seqlen, nheads)

        # Pad or truncate xBC seqlen to d_conv
        xBC = silu(
            self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)[:, : u.shape[1], :]
        )  # (batch, seqlen, d_inner + 2 * d_state))
        x, B, C = torch.split(
            xBC, [self.d_inner, self.d_state, self.d_state], dim=-1
        )

        _b, _l, _hp = x.shape
        _h = _hp // self.headdim
        _p = self.headdim
        x = x.reshape(_b, _l, _h, _p)

        y = self.ssd(x * dt.unsqueeze(-1),
                     A * dt,
                     B.unsqueeze(2),
                     C.unsqueeze(2), )

        y = y + x * self.D.unsqueeze(-1)

        _b, _l, _h, _p = y.shape
        y = y.reshape(_b, _l, _h * _p)

        y = self.norm(y, z)
        y = self.out_proj(y)

        return y

    def segsum(self, x: Tensor) -> Tensor:
        T = x.size(-1)
        device = x.device
        x = x[..., None].repeat(1, 1, 1, 1, T)
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=-1)
        x = x.masked_fill(~mask, 0)
        x_segsum = torch.cumsum(x, dim=-2)
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=0)
        x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
        return x_segsum

    def ssd(self, x, A, B, C):
        chunk_size = self.chunk_size
        # if x.shape[1] % chunk_size == 0:
        #
        x = x.reshape(x.shape[0], x.shape[1] // chunk_size, chunk_size, x.shape[2], x.shape[3], )
        B = B.reshape(B.shape[0], B.shape[1] // chunk_size, chunk_size, B.shape[2], B.shape[3], )
        C = C.reshape(C.shape[0], C.shape[1] // chunk_size, chunk_size, C.shape[2], C.shape[3], )
        A = A.reshape(A.shape[0], A.shape[1] // chunk_size, chunk_size, A.shape[2])
        A = A.permute(0, 3, 1, 2)
        A_cumsum = torch.cumsum(A, dim=-1)

        # 1. Compute the output for each intra-chunk (diagonal blocks)
        L = torch.exp(self.segsum(A))
        Y_diag = torch.einsum("bclhn, bcshn, bhcls, bcshp -> bclhp", C, B, L, x)

        # 2. Compute the state for each intra-chunk
        # (right term of low-rank factorization of off-diagonal blocks; B terms)
        decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
        states = torch.einsum("bclhn, bhcl, bclhp -> bchpn", B, decay_states, x)

        # 3. Compute the inter-chunk SSM recurrence; produces correct SSM states at chunk boundaries
        # (middle term of factorization of off-diag blocks; A terms)

        initial_states = torch.zeros_like(states[:, :1])
        states = torch.cat([initial_states, states], dim=1)

        decay_chunk = torch.exp(self.segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))[0]
        new_states = torch.einsum("bhzc, bchpn -> bzhpn", decay_chunk, states)
        states = new_states[:, :-1]

        # 4. Compute state -> output conversion per chunk
        # (left term of low-rank factorization of off-diagonal blocks; C terms)
        state_decay_out = torch.exp(A_cumsum)
        Y_off = torch.einsum("bclhn, bchpn, bhcl -> bclhp", C, states, state_decay_out)

        # Add output of intra-chunk and inter-chunk terms (diagonal and off-diagonal blocks)
        # Y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")
        Y = Y_diag + Y_off
        Y = Y.reshape(Y.shape[0], Y.shape[1] * Y.shape[2], Y.shape[3], Y.shape[4], )

        return Y


class _BiMamba2(nn.Module):
    def __init__(self,
                 cin: int,
                 cout: int,
                 d_model: int,  # model dimension (D)
                 n_layer: int = 24,  # number of Mamba-2 layers in the language model
                 d_state: int = 128,  # state dimension (N)
                 d_conv: int = 4,  # convolution kernel size
                 expand: int = 2,  # expansion factor (E)
                 headdim: int = 64,  # head dimension (P)
                 chunk_size: int = 64,  # matrix partition size (Q)
                 ):
        super().__init__()
        self.fc_in = nn.Linear(cin, d_model, bias=False)  # 调整通道数到cmid
        self.mamba2_for = Mamba2(d_model, n_layer, d_state, d_conv, expand, headdim, chunk_size, )  # 正向
        self.mamba2_back = Mamba2(d_model, n_layer, d_state, d_conv, expand, headdim, chunk_size, )  # 负向
        self.fc_out = nn.Linear(d_model, cout, bias=False)  # 调整通道数到cout
        self.chunk_size = chunk_size

    @abstractmethod
    def forward(self, x):
        pass


class BiMamba2_1D(_BiMamba2):
    def __init__(self, cin, cout, d_model, **mamba2_args):
        super().__init__(cin, cout, d_model, **mamba2_args)

    def forward(self, x):
        l = x.shape[2]
        x = F.pad(x, (0, (64 - x.shape[2] % 64) % 64))  # 将 l , pad到4的倍数, [b, c64,l4]
        x = x.transpose(1, 2)  # 转成 1d 信号 [b, c64, d4*w4*h4]
        x = self.fc_in(x)  # 调整通道数为目标通道数
        x1 = self.mamba2_for(x)
        x2 = self.mamba2_back(x.flip(1)).flip(1)
        x = x1 + x2
        x = self.fc_out(x)  # 调整通道数为目标通道数
        x = x.transpose(1, 2)  # 转成 1d 信号 [b, c64, d4*w4*h4] ]
        x = x[:, :, :l]  # 截取原图大小
        return x


class BiMamba2_2D(_BiMamba2):
    def __init__(self, cin, cout, d_model, **mamba2_args):
        super().__init__(cin, cout, d_model, **mamba2_args)

    def forward(self, x):
        h, w = x.shape[2:]
        x = F.pad(x, (0, (8 - x.shape[3] % 8) % 8,
                      0, (8 - x.shape[2] % 8) % 8)
                  )  # 将 h , w  pad到8的倍数, [b, c64, h8, w8]
        _b, _c, _h, _w = x.shape
        x = x.permute(0, 2, 3, 1).reshape(_b, _h * _w, _c)
        x = self.fc_in(x)  # 调整通道数为目标通道数
        x1 = self.mamba2_for(x)
        x2 = self.mamba2_back(x.flip(1)).flip(1)
        x = x1 + x2
        x = self.fc_out(x)  # 调整通道数为目标通道数
        x = x.reshape(_b, _h, _w, -1, )
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(_b, -1, _h, _w, )
        x = x[:, :, :h, :w]  # 截取原图大小
        return x


class BiMamba2_3D(_BiMamba2):
    def __init__(self, cin, cout, d_model, **mamba2_args):
        super().__init__(cin, cout, d_model, **mamba2_args)

    def forward(self, x):
        d, h, w = x.shape[2:]
        x = F.pad(x, (0, (4 - x.shape[4] % 4) % 4,
                      0, (4 - x.shape[3] % 4) % 4,
                      0, (4 - x.shape[2] % 4) % 4)
                  )  # 将 d, h, w , pad到4的倍数, [b, c64,d4, h4, w4]
        _b, _c, _d, _h, _w = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(_b, _d * _h * _w, _c)
        x = self.fc_in(x)  # 调整通道数为目标通道数
        x1 = self.mamba2_for(x)
        x2 = self.mamba2_back(x.flip(1)).flip(1)
        x = x1 + x2
        x = self.fc_out(x)  # 调整通道数为目标通道数
        x = x.reshape(_b, _d, _h, _w, -1)
        x = x.permute(0, 4, 1, 2, 3)
        x=x.reshape(_b, -1, _d, _h, _w, )
        x = x[:, :, :d, :h, :w]  # 截取原图大小
        return x


class BiMamba2(_BiMamba2):
    def __init__(self, cin, cout, d_model, **mamba2_args):
        super().__init__(cin, cout, d_model, **mamba2_args)

    def forward(self, x):
        size = x.shape[2:]
        out_size = list(x.shape)
        out_size[1] = -1

        x = torch.flatten(x, 2)  # b c size
        l = x.shape[2]
        _s = self.chunk_size
        x = F.pad(x, [0, (_s - x.shape[2] % _s) % _s])  # 将 l, pad到chunk_size的倍数, [b, c64,l4]
        x = x.transpose(1, 2)  # 转成 1d 信号
        x = self.fc_in(x)  # 调整通道数为目标通道数
        x1 = self.mamba2_for(x)
        x2 = self.mamba2_back(x.flip(1)).flip(1)
        x = x1 + x2
        x = self.fc_out(x)  # 调整通道数为目标通道数
        x = x.transpose(1, 2)  # 转成 1d 信号
        x = x[:, :, :l]  # 截取原图大小
        x = x.reshape(out_size)

        return x


def test_export_jit_script(net, x):
    y = net(x)
    net_script = torch.jit.script(net)
    torch.jit.save(net_script, 'net.jit.script')
    net2 = torch.jit.load('net.jit.script')
    y = net2(x)
    print(y.shape)


def test_export_onnx(net, x):
    torch.onnx.export(net,
                      x,
                      "net.onnx",  # 输出的 ONNX 文件名
                      export_params=True,  # 存储训练参数
                      opset_version=14,  # 指定 ONNX 操作集版本
                      do_constant_folding=False,  # 是否执行常量折叠优化
                      input_names=['input'],  # 输入张量的名称
                      output_names=['output'],  # 输出张量的名称
                      dynamic_axes={'input': {0: 'batch_size'},  # 可变维度的字典
                                    'output': {0: 'batch_size'}})


if __name__ == '__main__':
    # 通用的多维度双向mamba2
    from torchnssd import (
        export_jit_script,
        export_onnx,
        statistics,
        test_run,
    )

    net_n = BiMamba2_1D(61, 128, 32).cuda()
    net_n.eval()
    x = torch.randn(1, 61, 63).cuda()
    export_jit_script(net_n)
    export_onnx(net_n, x)
    test_run(net_n, x)
    statistics(net_n, (61, 63))
