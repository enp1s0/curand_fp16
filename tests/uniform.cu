#include <iostream>
#include <curand_fp16/curand_fp16.hpp>

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

	std::printf("avg = %e, var = %e\n", avg, var);

	for (unsigned i = 0; i < 10; i++) {
		std::printf("%e\n", __half2float(ptr[i]));
	}

	mtk::curand_fp16::destroy(generator);
	cudaFree(ptr);
}

int main() {
	test_curand_fp16(1u << 20, CURAND_RNG_PSEUDO_PHILOX4_32_10);
}
