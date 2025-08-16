import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from conformer import Conformer

class NeuralModel(nn.Module):
    """
    Принимает |X| STFT: (B, C, F, T_spec) и предсказывает комплексные маски
    в свернутом виде: (B, 2 * (sources*channels), F, T_spec)
    где 2 — это [real, imag].
    """
    def __init__(
        self,
        in_channels: int = 2,
        sources: int = 2,
        freq_bins: int = 2049,
        embed_dim: int = 512,
        depth: int = 8,
        dim_head: int = 64,
        heads: int = 8,
        ff_mult: int = 4,
        conv_expansion_factor: int = 2,
        conv_kernel_size: int = 31,
        attn_dropout: float = 0.1,
        ff_dropout: float = 0.1,
        conv_dropout: float = 0.1,
    ):
        super().__init__()
        self.freq_bins = freq_bins
        self.in_channels = in_channels
        self.sources = sources
        self.out_masks = sources * in_channels
        self.embed_dim = embed_dim

        self.input_proj_stft = nn.Linear(freq_bins * in_channels, embed_dim)
        self.model = Conformer(
            dim=embed_dim,
            depth=depth,
            dim_head=dim_head,
            heads=heads,
            ff_mult=ff_mult,
            conv_expansion_factor=conv_expansion_factor,
            conv_kernel_size=conv_kernel_size,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            conv_dropout=conv_dropout,
        )
        # 2 = [real, imag]
        self.output_proj = nn.Linear(embed_dim, freq_bins * self.out_masks * 2)

    def forward(self, x_stft_mag: torch.Tensor) -> torch.Tensor:
        """
        x_stft_mag: (B, C, F, T_spec)
        returns: (B, 2 * (sources*channels), F, T_spec)  — real/imag масок
        """
        assert x_stft_mag.dim() == 4, f"Expected (B,C,F,T), got {tuple(x_stft_mag.shape)}"
        B, C, F, T_spec = x_stft_mag.shape
        # (B, T_spec, C*F)
        x_stft_mag = x_stft_mag.permute(0, 3, 1, 2).contiguous().view(B, T_spec, C * F)

        x = self.input_proj_stft(x_stft_mag)     # (B, T_spec, E)
        x = self.model(x)                        # (B, T_spec, E)
        x = torch.tanh(x)                        # стабилизируем
        x = self.output_proj(x)                  # (B, T_spec, F * out_masks * 2)

        # back to (B, 2*out_masks, F, T_spec)
        x = x.reshape(B, T_spec, self.out_masks * 2, F).permute(0, 2, 3, 1).contiguous()
        return x


class ConformerMSS(nn.Module):
    """
    Совместимо с твоим train:
      forward(x: (B, C, T)) -> y_hat: (B, S, C, T)
    где S = число источников (sources).
    Внутри: STFT -> NeuralModel -> комплексные маски -> iSTFT.
    """
    def __init__(
        self,
        core: NeuralModel,
        n_fft: int = 4096,
        hop_length: int = 1024,
        win_length: Optional[int] = None,
        center: bool = True,
    ):
        super().__init__()
        self.core = core
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length if win_length is not None else n_fft
        self.center = center

        window = torch.hann_window(self.win_length)
        # окно — буфер, чтобы таскалось на .to(device)
        self.register_buffer("window", window, persistent=False)

        # sanity-check: freq_bins у core должен совпадать с n_fft//2 + 1
        expected_bins = n_fft // 2 + 1
        assert core.freq_bins == expected_bins, (
            f"NeuralModel.freq_bins={core.freq_bins} != n_fft//2+1={expected_bins}. "
            f"Поставь freq_bins={expected_bins} при создании core."
        )

    def _stft(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T) -> spec: complex (B, C, F, TT)
        """
        assert x.dim() == 3, f"Expected (B,C,T), got {tuple(x.shape)}"
        B, C, T = x.shape
        x_bc_t = x.reshape(B * C, T)
        spec = torch.stft(
            x_bc_t,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(x.device),
            center=self.center,
            return_complex=True,
        )  # (B*C, F, TT)
        F, TT = spec.shape[-2], spec.shape[-1]
        spec = spec.reshape(B, C, F, TT)
        return spec

    def _istft(self, spec: torch.Tensor, length: int) -> torch.Tensor:
        """
        spec: complex (B, C, F, TT) -> audio: (B, C, T)
        """
        B, C, F, TT = spec.shape
        spec_bc = spec.reshape(B * C, F, TT)
        y_bc_t = torch.istft(
            spec_bc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(spec.device),
            center=self.center,
            length=length,
        )
        return y_bc_t.reshape(B, C, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T)  (микс в волне)
        returns y_hat: (B, S, C, T) — предсказанные источники в волне
        """
        B, C, T = x.shape
        # 1) STFT
        mix_spec = self._stft(x)                     # (B, C, F, TT)
        mix_mag = mix_spec.abs()                     # (B, C, F, TT)

        # 2) Прогон через core -> real/imag масок
        mask_ri = self.core(mix_mag)                 # (B, 2*(S*C), F, TT2)
        _, two_sc, F, TT2 = mask_ri.shape

        S = self.core.sources
        assert two_sc == 2 * (S * C), (
            f"core вернул {two_sc} каналов масок, ожидалось {2*(S*C)} "
            f"(2*[real/imag]*[sources*channels]). Проверь in_channels/sources."
        )

        # 3) Синхронизация по времени (если вдруг TT != TT2)
        TT = mix_spec.shape[-1]
        TT_min = min(TT, TT2)
        if TT != TT_min:
            mix_spec = mix_spec[..., :TT_min]
        if TT2 != TT_min:
            mask_ri = mask_ri[..., :TT_min]
        TT = TT_min
        # теперь у обоих время = TT

        # 4) Преобразуем к (B, 2, S, C, F, TT)
        mask_ri = mask_ri.view(B, 2, S, C, F, TT).contiguous()
        mask_real = mask_ri[:, 0]                    # (B, S, C, F, TT)
        mask_imag = mask_ri[:, 1]                    # (B, S, C, F, TT)
        masks_c = torch.complex(mask_real, mask_imag)

        # 5) Применяем маски к комплексному спектру микса
        mix_spec_bc = mix_spec.unsqueeze(1)          # (B, 1, C, F, TT)
        est_specs = masks_c * mix_spec_bc            # (B, S, C, F, TT)

        # 6) iSTFT по каждому источнику
        outs = []
        for s in range(S):
            y_s = self._istft(est_specs[:, s], length=T)  # (B, C, T)
            outs.append(y_s)
        y_hat = torch.stack(outs, dim=1)             # (B, S, C, T)
        return y_hat