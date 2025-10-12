import torch
import torch.nn as nn

from .layers import (get_norm)

class TFC(nn.Module):
    def __init__(self, c_in, c_out, l, k, bn_norm):
        super(TFC, self).__init__()

        self.H = nn.ModuleList()
        for i in range(l):
            if i == 0:
                c_in = c_in
            else:
                c_in = c_out
            self.H.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=k, stride=1, padding=k // 2),
                    get_norm(bn_norm, c_out),
                    nn.ReLU(),
                )
            )

    def forward(self, x):
        for h in self.H:
            x = h(x)
        return x


class DenseTFC(nn.Module):
    def __init__(self, c_in, c_out, l, k, bn_norm):
        super(DenseTFC, self).__init__()

        self.conv = nn.ModuleList()
        for i in range(l):
            self.conv.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=k, stride=1, padding=k // 2),
                    get_norm(bn_norm, c_out),
                    nn.ReLU(),
                )
            )

    def forward(self, x):
        for layer in self.conv[:-1]:
            x = torch.cat([layer(x), x], 1)
        return self.conv[-1](x)


class TFC_TDF(nn.Module):
    def __init__(self, c_in, c_out, l, f, k, bn, bn_norm, dense=False, bias=True):

        super(TFC_TDF, self).__init__()

        self.use_tdf = bn is not None

        self.tfc = DenseTFC(c_in, c_out, l, k, bn_norm) if dense else TFC(c_in, c_out, l, k, bn_norm)

        if self.use_tdf:
            if bn == 0:
                # print(f"TDF={f},{f}")
                self.tdf = nn.Sequential(
                    nn.Linear(f, f, bias=bias),
                    get_norm(bn_norm, c_out),
                    nn.ReLU()
                )
            else:
                # print(f"TDF={f},{f // bn},{f}")
                self.tdf = nn.Sequential(
                    nn.Linear(f, f // bn, bias=bias),
                    get_norm(bn_norm, c_out),
                    nn.ReLU(),
                    nn.Linear(f // bn, f, bias=bias),
                    get_norm(bn_norm, c_out),
                    nn.ReLU()
                )

    def forward(self, x):
        x = self.tfc(x)
        return x + self.tdf(x) if self.use_tdf else x


class TFC_TDF_Res1(nn.Module):
    def __init__(self, c_in, c_out, l, f, k, bn, bn_norm, dense=False, bias=True):

        super(TFC_TDF_Res1, self).__init__()

        self.use_tdf = bn is not None

        self.tfc = DenseTFC(c_in, c_out, l, k, bn_norm) if dense else TFC(c_in, c_out, l, k, bn_norm)

        self.res = TFC(c_in, c_out, 1, k, bn_norm)

        if self.use_tdf:
            if bn == 0:
                # print(f"TDF={f},{f}")
                self.tdf = nn.Sequential(
                    nn.Linear(f, f, bias=bias),
                    get_norm(bn_norm, c_out),
                    nn.ReLU()
                )
            else:
                # print(f"TDF={f},{f // bn},{f}")
                self.tdf = nn.Sequential(
                    nn.Linear(f, f // bn, bias=bias),
                    get_norm(bn_norm, c_out),
                    nn.ReLU(),
                    nn.Linear(f // bn, f, bias=bias),
                    get_norm(bn_norm, c_out),
                    nn.ReLU()
                )

    def forward(self, x):
        res = self.res(x)
        x = self.tfc(x)
        x = x + res
        return x + self.tdf(x) if self.use_tdf else x


class TFC_TDF_Res2(nn.Module):
    def __init__(self, c_in, c_out, l, f, k, bn, bn_norm, dense=False, bias=True):

        super(TFC_TDF_Res2, self).__init__()

        self.use_tdf = bn is not None

        self.tfc1 = TFC(c_in, c_out, l, k, bn_norm)
        self.tfc2 = TFC(c_in, c_out, l, k, bn_norm)

        self.res = TFC(c_in, c_out, 1, k, bn_norm)

        if self.use_tdf:
            if bn == 0:
                # print(f"TDF={f},{f}")
                self.tdf = nn.Sequential(
                    nn.Linear(f, f, bias=bias),
                    get_norm(bn_norm, c_out),
                    nn.ReLU()
                )
            else:
                # print(f"TDF={f},{f // bn},{f}")
                self.tdf = nn.Sequential(
                    nn.Linear(f, f // bn, bias=bias),
                    get_norm(bn_norm, c_out),
                    nn.ReLU(),
                    nn.Linear(f // bn, f, bias=bias),
                    get_norm(bn_norm, c_out),
                    nn.ReLU()
                )

    def forward(self, x):
        res = self.res(x)
        x = self.tfc1(x)
        if self.use_tdf:
            x = x + self.tdf(x)
        x = self.tfc2(x)
        x = x + res
        return x
