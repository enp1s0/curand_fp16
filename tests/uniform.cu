#include <iostream>
#include <chrono>
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

void test_throughput(
		const curandRngType_t rng
		) {
	std::printf("# %s [%s]\n", __func__, get_curand_rng_name_str(rng));
	constexpr std::size_t min_N = 1u << 10;
	constexpr std::size_t max_N = 1u << 30;
	constexpr std::size_t num_run = 1u << 10;
	half* ptr;
	cudaMallocManaged(&ptr, sizeof(half) * max_N);

	mtk::curand_fp16::generator_t generator;
	mtk::curand_fp16::create(generator, rng);
	mtk::curand_fp16::set_seed(generator, 0);

	for (std::size_t N = min_N; N <= max_N; N <<= 1) {
		cudaDeviceSynchronize();
		const auto start_clock = std::chrono::system_clock::now();
		for (std::size_t i = 0; i < num_run; i++) {
			mtk::curand_fp16::uniform(generator, ptr, N);
		}
		cudaDeviceSynchronize();
		const auto end_clock = std::chrono::system_clock::now();
		const auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end_clock - start_clock).count() * 1e-6 / num_run;
		std::printf("%10lu, %.3e [s], %.3e [GB/s]\n",
				N,
				elapsed_time,
				N * sizeof(half) / elapsed_time * 1e-9
				);
	}

	mtk::curand_fp16::destroy(generator);
	cudaFree(ptr);
}

int main() {
	test_curand_fp16(1u << 20, CURAND_RNG_PSEUDO_MRG32K3A     );
	test_curand_fp16(1u << 20, CURAND_RNG_PSEUDO_XORWOW       );
	test_curand_fp16(1u << 20, CURAND_RNG_PSEUDO_PHILOX4_32_10);
	test_throughput(CURAND_RNG_PSEUDO_MRG32K3A     );
	test_throughput(CURAND_RNG_PSEUDO_XORWOW       );
	test_throughput(CURAND_RNG_PSEUDO_PHILOX4_32_10);
}
