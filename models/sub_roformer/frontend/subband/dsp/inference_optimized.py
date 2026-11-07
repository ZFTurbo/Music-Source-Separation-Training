from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from einops import rearrange

from .analytic import real_to_analytic
from .convolve import fftconvolve
from .subband_fast_triangle import SubbandFilter
from .utils import shift_frequency


OPTIMIZATION_STEPS = {
    1: "Validate fixed filters once instead of every band group",
    2: "Reuse the analysis input FFT across band groups",
    3: "Cache analysis and synthesis filter FFTs",
    4: "Cache analysis and synthesis modulation tables",
}


class InferenceSubbandFilter(SubbandFilter):
    """Inference-only DSP replacement; parameters and state-dict keys are unchanged."""

    def __init__(self, source: SubbandFilter, level: int):
        torch.nn.Module.__init__(self)
        for name in (
            "sr", "banks", "window_type", "factor", "chunk_size",
            "bandpass_filter_len", "upsample_filter_len",
        ):
            setattr(self, name, getattr(source, name))
        for name in ("w", "f_center", "up"):
            self.register_buffer(name, getattr(source, name))
        self.level = level
        self._cache_key = None
        self._filter_spectra = {}
        self._modulations = {}
        self.register_load_state_dict_post_hook(self._after_load)

    def _after_load(self, module, incompatible_keys):
        self.clear_cache()

    def clear_cache(self):
        self._cache_key = None
        self._filter_spectra.clear()
        self._modulations.clear()

    def cache_bytes(self):
        return sum(
            value.numel() * value.element_size()
            for cache in (self._filter_spectra, self._modulations)
            for value in cache.values()
        )

    def _apply(self, function, recurse=True):
        self.clear_cache()
        return super()._apply(function, recurse=recurse)

    def train(self, mode=True):
        if mode:
            self.clear_cache()
        return super().train(mode)

    def _prepare_cache(self, samples, inputs):
        key = (
            samples, inputs.device, inputs.dtype, self.factor, self.chunk_size, self.sr,
            tuple((id(value), value._version) for value in (self.w, self.up, self.f_center)),
        )
        if key != self._cache_key:
            self.clear_cache()
            assert torch.all(self.w[:, 0] == 0)
            self._cache_key = key

    def _shift(self, inputs, frequencies, stride, key):
        if self.level < 4:
            return shift_frequency(inputs, frequencies, self.sr, stride)
        modulation = self._modulations.get(key)
        if modulation is None:
            if frequencies.dtype in {torch.float16, torch.bfloat16}:
                frequencies = frequencies.float()
            omega = frequencies / (self.sr / 2) * math.pi
            positions = torch.arange(inputs.shape[-1], device=inputs.device) * stride
            modulation = torch.exp(1.j * omega[:, None] * positions[None, :])
            self._modulations[key] = modulation
        return inputs * modulation

    def analysis(self, inputs):
        if self.training or torch.is_grad_enabled() or any(
            torch.is_inference(value) for value in (self.w, self.up, self.f_center)
        ):
            return super().analysis(inputs)
        batch_size = inputs.shape[0]
        inputs = real_to_analytic(inputs)
        self._prepare_cache(inputs.shape[-1], inputs)
        inputs = rearrange(inputs, "b c l -> (b c) l")
        samples = inputs.shape[-1] // self.factor
        output = inputs.new_empty(inputs.shape[0], len(self.banks), samples)
        phases = rearrange(inputs, "b (t1 t2) -> b t2 t1", t2=self.factor)
        filter_samples = self.w.shape[-1] // self.factor
        fft_samples = samples + filter_samples - 1
        input_spectrum = None
        if self.level >= 2:
            input_spectrum = torch.fft.fft(F.pad(phases, (0, fft_samples - samples)))
        for start in range(0, len(self.banks), self.chunk_size):
            weights = self.w[start:start + self.chunk_size]
            weights = rearrange(weights, "k (n1 n2) -> k n2 n1", n2=self.factor)
            assert filter_samples % 2 == 0
            weights = torch.flip(weights, dims=[2])
            if input_spectrum is None:
                filtered = fftconvolve(phases, weights, use_complex_fft=True, mode="full")
            else:
                filter_spectrum = self._filter_spectra.get(("analysis", start))
                if filter_spectrum is None:
                    if weights.dtype in {torch.float16, torch.bfloat16}:
                        weights = weights.float()
                    filter_spectrum = torch.fft.fft(F.pad(weights, (0, fft_samples - filter_samples)))
                    if self.level >= 3:
                        self._filter_spectra[("analysis", start)] = filter_spectrum
                product = torch.einsum("bil,oil->bol", input_spectrum, filter_spectrum)
                filtered = torch.fft.ifft(product, n=fft_samples)
            filtered = filtered[..., filter_samples // 2 - 1:filter_samples // 2 - 1 + samples]
            filtered = self._shift(
                filtered, -self.f_center[start:start + self.chunk_size], self.factor, ("analysis", start),
            )
            output[:, start:start + self.chunk_size] = filtered
        return rearrange(output, "(b c) k l -> b c k l", b=batch_size)


    def synthesis(self, inputs):
        if self.training or torch.is_grad_enabled() or self.level < 3 or any(
            torch.is_inference(value) for value in (self.w, self.up, self.f_center)
        ):
            return super().synthesis(inputs)
        batch_size = inputs.shape[0]
        inputs = rearrange(inputs, "b c k l -> (b c) k l")
        samples = inputs.shape[-1] * self.factor
        self._prepare_cache(samples, inputs)
        weights = rearrange(self.up, "(n1 n2) -> n2 1 n1", n2=self.factor)
        fft_samples = inputs.shape[-1] + weights.shape[-1] - 1
        filter_spectrum = self._filter_spectra.get(("synthesis", 0))
        if filter_spectrum is None:
            if weights.dtype in {torch.float16, torch.bfloat16}:
                weights = weights.float()
            filter_spectrum = torch.fft.fft(F.pad(weights, (0, fft_samples - weights.shape[-1])))
            self._filter_spectra[("synthesis", 0)] = filter_spectrum
        output = inputs.new_zeros(inputs.shape[0], samples)
        for start in range(0, len(self.banks), self.chunk_size):
            chunk = rearrange(inputs[:, start:start + self.chunk_size], "b s l -> (b s) l")
            chunk = rearrange(chunk * self.factor, "b l -> b 1 l")
            spectrum = torch.fft.fft(F.pad(chunk, (0, fft_samples - chunk.shape[-1])))
            product = torch.einsum("bil,oil->bol", spectrum, filter_spectrum)
            expanded = torch.fft.ifft(product, n=fft_samples)
            expanded = rearrange(expanded, "b t2 t1 -> b (t1 t2)")
            expanded = expanded[..., self.up.shape[-1] // 2:self.up.shape[-1] // 2 + samples]
            expanded = rearrange(expanded, "(b s) l -> b s l", b=inputs.shape[0])
            expanded = self._shift(expanded, self.f_center[start:start + self.chunk_size], 1, ("synthesis", start))
            output.add_(expanded.sum(dim=1))
        return rearrange(output, "(b c) l -> b c l", b=batch_size).real
