"""Retain the model's FFT cache on MPS without reverting DSP3 modulation."""

from ..dsp.inference_optimized import InferenceSubbandFilter as CachedSubbandFilter

from . import subband_fast_triangle
from .subband_fast_triangle import SubbandFilter


class InferenceSubbandFilter(CachedSubbandFilter, SubbandFilter):
    """Reuse cache invalidation and FFTs; uncached paths resolve to DSP3 via MRO."""

    def __init__(self, source: SubbandFilter):
        super().__init__(source, level=3)

    def _shift(self, inputs, frequencies, stride, key):
        return subband_fast_triangle.shift_frequency(inputs, frequencies, self.sr, stride)

    def analysis(self, inputs):
        if inputs.device.type != "mps":
            return SubbandFilter.analysis(self, inputs)
        return super().analysis(inputs)

    def synthesis(self, inputs):
        if inputs.device.type != "mps":
            return SubbandFilter.synthesis(self, inputs)
        return super().synthesis(inputs)
