#include <metal_stdlib>
using namespace metal;

kernel void quantize_frequencies(
    device const float* frequencies,
    device long* steps,
    constant ulong& sample_rate,
    uint band [[thread_position_in_grid]]) {
    float frequency = frequencies[band];
    long scaled = long(rint(abs(frequency) * 4294967296.0f));
    long magnitude = (scaled + long(sample_rate / 2)) / long(sample_rate);
    steps[band] = frequency < 0 ? -magnitude : magnitude;
}

kernel void shift_frequency(
    device const float2* source,
    device const long* steps,
    device float2* target,
    constant ulong& length,
    constant ulong& bands,
    constant ulong& band_stride,
    constant ulong& batch_stride,
    constant ulong& stride,
    constant ulong& conjugate,
    uint3 position [[thread_position_in_grid]]) {
    uint step = uint(steps[position.y]);
    uint phase_word = step * position.x * uint(stride);
    float phase = float(as_type<int>(phase_word)) * 1.4629180792671596e-9f;
    float real_oscillator = precise::cos(phase);
    float imag_oscillator = precise::sin(phase);
    if (conjugate != 0) {
        imag_oscillator = -imag_oscillator;
    }
    ulong source_offset = position.z * batch_stride + position.y * band_stride + position.x;
    ulong offset = (position.z * bands + position.y) * length + position.x;
    float2 sample = source[source_offset];
    target[offset] = float2(sample.x * real_oscillator - sample.y * imag_oscillator,
                           sample.x * imag_oscillator + sample.y * real_oscillator);
}
