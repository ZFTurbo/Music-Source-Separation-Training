#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec/vec.h>
#include <torch/library.h>
#include <cmath>
#include <cstdint>
#include <type_traits>

template <typename input_t, typename real_t>
void modulate(const at::Tensor& input, const at::Tensor& steps,
              at::Tensor& output, int64_t stride, bool conjugate) {
    const auto* input_data = input.const_data_ptr<input_t>();
    const auto* step_data = steps.const_data_ptr<int64_t>();
    auto* output_data = output.mutable_data_ptr<c10::complex<real_t>>();
    const int64_t length = input.size(-1);
    const int64_t bands = steps.numel();
    const int64_t batches = input.numel() / (bands * length);
    const int64_t band_stride = input.stride(-2);
    const int64_t batch_stride = input.dim() == 3 ? input.stride(-3) : bands * length;
    constexpr int64_t tile_size = 1024;
    const int64_t tiles = (length + tile_size - 1) / tile_size;
    using Vector = at::vec::Vectorized<real_t>;
    const real_t radians_per_word = static_cast<real_t>(6.283185307179586476925286766559 / 4294967296.0);
    at::parallel_for(0, bands * tiles, 4, [&](int64_t begin, int64_t end) {
        alignas(64) real_t phases[tile_size];
        alignas(64) real_t cosines[tile_size];
        alignas(64) real_t sines[tile_size];
        for (int64_t tile = begin; tile < end; ++tile) {
            const int64_t band = tile / tiles;
            const int64_t start = (tile % tiles) * tile_size;
            const int64_t count = std::min(tile_size, length - start);
            const uint32_t increment = static_cast<uint32_t>(step_data[band]) * static_cast<uint32_t>(stride);
            for (int64_t sample = 0; sample < count; ++sample) {
                const uint32_t word = increment * static_cast<uint32_t>(start + sample);
                const int32_t signed_word = c10::bit_cast<int32_t>(word);
                phases[sample] = static_cast<real_t>(signed_word) * radians_per_word;
            }
            for (int64_t sample = 0; sample < count; sample += Vector::size()) {
                const int64_t width = std::min(int64_t(Vector::size()), count - sample);
                const auto phase = Vector::loadu(phases + sample, width);
                phase.cos().store(cosines + sample, width);
                const auto sine = phase.sin();
                (conjugate ? sine.neg() : sine).store(sines + sample, width);
            }
            for (int64_t batch = 0; batch < batches; ++batch) {
                const int64_t offset = (batch * bands + band) * length + start;
                const int64_t source_offset = batch * batch_stride + band * band_stride + start;
                for (int64_t sample = 0; sample < count; ++sample) {
                    const real_t cosine = cosines[sample];
                    const real_t sine = sines[sample];
                    if constexpr (std::is_same_v<input_t, real_t>) {
                        output_data[offset + sample] = c10::complex<real_t>(input_data[source_offset + sample] * cosine, input_data[source_offset + sample] * sine);
                    } else {
                        const real_t real = input_data[source_offset + sample].real();
                        const real_t imag = input_data[source_offset + sample].imag();
                        output_data[offset + sample] = c10::complex<real_t>(real * cosine - imag * sine, real * sine + imag * cosine);
                    }
                }
            }
        }
    });
}

at::Tensor shift_frequency_cpu(const at::Tensor& input, const at::Tensor& steps, int64_t stride, bool conjugate) {
    TORCH_CHECK(input.device().is_cpu() && steps.device().is_cpu(), "DSP3 CPU kernel requires CPU tensors");
    TORCH_CHECK(input.dim() >= 2 && (input.is_contiguous() || (input.dim() <= 3 && input.stride(-1) == 1)), "DSP3 CPU input must be contiguous or row-contiguous with two or three dimensions");
    TORCH_CHECK(!input.is_conj() && !input.is_neg(), "DSP3 CPU input must have resolved conjugate and negative view bits");
    TORCH_CHECK(steps.dim() == 1 && steps.scalar_type() == at::kLong && steps.is_contiguous(), "DSP3 phase steps must be contiguous one-dimensional int64");
    TORCH_CHECK(input.size(-2) == steps.numel(), "DSP3 input band dimension must match phase steps");
    const auto dtype = input.scalar_type();
    TORCH_CHECK(dtype == at::kFloat || dtype == at::kDouble || dtype == at::kComplexFloat || dtype == at::kComplexDouble, "DSP3 CPU kernel supports float32, float64, complex64, and complex128");
    const auto output_dtype = dtype == at::kDouble || dtype == at::kComplexDouble ? at::kComplexDouble : at::kComplexFloat;
    auto output = at::empty(input.sizes(), input.options().dtype(output_dtype));
    if (input.numel() == 0) {
        return output;
    }
    switch (dtype) {
        case at::kFloat: modulate<float, float>(input, steps, output, stride, conjugate); break;
        case at::kDouble: modulate<double, double>(input, steps, output, stride, conjugate); break;
        case at::kComplexFloat: modulate<c10::complex<float>, float>(input, steps, output, stride, conjugate); break;
        case at::kComplexDouble: modulate<c10::complex<double>, double>(input, steps, output, stride, conjugate); break;
        default: TORCH_CHECK(false, "Unsupported DSP3 input dtype");
    }
    return output;
}

TORCH_LIBRARY_FRAGMENT(subroformer_dsp3_native, library) {
    library.def("shift_frequency_cpu(Tensor input, Tensor steps, int stride=1, bool conjugate=False) -> Tensor");
}

TORCH_LIBRARY_IMPL(subroformer_dsp3_native, CPU, library) {
    library.impl("shift_frequency_cpu", &shift_frequency_cpu);
}
