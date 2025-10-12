import torch
import torch.nn as nn

class DTTNetBase(nn.Module):
    def __init__(self,
                 target_name,
                 dim_f, dim_t, n_fft, hop_length, overlap,
                 audio_ch,
                 **kwargs):
        super().__init__()
        self.target_name = target_name
        self.dim_c_in = audio_ch * 2
        self.dim_c_out = audio_ch * 2
        self.dim_f = dim_f
        self.dim_t = dim_t
        self.n_fft = n_fft
        self.n_bins = n_fft // 2 + 1
        self.hop_length = hop_length
        self.audio_ch = audio_ch
        self.overlap = overlap

        # The chunk size your model will expect for its forward pass
        self.chunk_size = hop_length * (self.dim_t - 1)

        # Registering buffers instead of nn.Parameter for non-trainable tensors
        self.register_buffer('window', torch.hann_window(window_length=self.n_fft, periodic=True))
        self.register_buffer('freq_pad', torch.zeros([1, self.dim_c_out, self.n_bins - self.dim_f, 1]))

    def stft(self, x):
        dim_b = x.shape[0]
        x = x.reshape([dim_b * self.audio_ch, -1])

        # 1. Force complex output (Required by modern PyTorch)
        complex_spec = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window, center=True, return_complex=True)

        # 2. Separate complex tensor into 2 real channels (Real/Imaginary)
        x_real = complex_spec.real.unsqueeze(1)
        x_imag = complex_spec.imag.unsqueeze(1)
        x = torch.cat([x_real, x_imag], dim=1) # (batch*c, 2, n_bins, dim_t)

        # 3. Reshape back to the expected input format (batch, c*2, n_bins, dim_t)
        x = x.reshape([dim_b, self.audio_ch * 2, self.n_bins, -1])

        # 4. Slice the frequency dimension to the desired size
        return x[:, :, :self.dim_f]

    def istft(self, x):
        dim_b = x.shape[0]

        # 1. Pad frequency axis
        x = torch.cat([x, self.freq_pad.repeat([x.shape[0], 1, 1, x.shape[-1]])], -2)

        # 2. Reshape from (batch, c*2, n_bins, dim_t) back to (batch*c, 2, n_bins, dim_t)
        x = x.reshape([dim_b * self.audio_ch, 2, self.n_bins, -1])

        # 3. Combine 2 real channels into a single complex tensor
        complex_spec = torch.complex(x[:, 0], x[:, 1])

        # 4. Perform ISTFT
        x = torch.istft(complex_spec, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window, center=True)

        return x.reshape([dim_b, self.audio_ch, -1])