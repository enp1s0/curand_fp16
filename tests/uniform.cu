#include <iostream>
#include <curand_fp16/curand_fp16.hpp>

namespace {
const char* get_curand_rng_name_str(const curandRngType_t rng) {
	switch (rng) {
	case CURAND_RNG_PSEUDO_XORWOW:
		return "XORWOW";
	case CURAND_RNG_PSEUDO_MRG32K3A:
		return "MRG32K3A";
	case CURAND_RNG_PSEUDO_PHILOX4_32_10:
		return "PHILOX4_32_10";
	default:
		return "Unknown";
	}
}
} // noname namespace

void test_curand_fp16(
		const std::size_t N,
		curandRngType_t rng
		) {
	half* ptr;
	cudaMallocManaged(&ptr, sizeof(half) * N);

	mtk::curand_fp16::generator_t generator;
	mtk::curand_fp16::create(generator, rng);
	mtk::curand_fp16::set_seed(generator, 0);

	mtk::curand_fp16::uniform(generator, ptr, N);
	cudaDeviceSynchronize();

	double sum = 0;
	for (std::size_t i = 0; i < N; i++) {
		sum += __half2float(ptr[i]);
	}
	const double avg = sum / N;
	double tmp = 0;
	for (std::size_t i = 0; i < N; i++) {
		const auto diff = avg - __half2float(ptr[i]);
		tmp += diff * diff;
	}
	const auto var = tmp / (N - 1);

	std::printf("[%15s] avg = %e [theo = 1/2], var = %e [theo = 1/12]\n", get_curand_rng_name_str(rng), avg, var);

	mtk::curand_fp16::destroy(generator);
	cudaFree(ptr);
}

int main() {
	test_curand_fp16(1u << 20, CURAND_RNG_PSEUDO_MRG32K3A     );
	test_curand_fp16(1u << 20, CURAND_RNG_PSEUDO_XORWOW       );
	test_curand_fp16(1u << 20, CURAND_RNG_PSEUDO_PHILOX4_32_10);
}
