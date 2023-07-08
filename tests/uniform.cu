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
		curandRngType_t rng,
		const bool pm
		) {
	half* ptr;
	cudaMallocManaged(&ptr, sizeof(half) * N);

	mtk::curand_fp16::generator_t generator;
	mtk::curand_fp16::create(generator, rng);
	mtk::curand_fp16::set_seed(generator, 0);

	mtk::curand_fp16::uniform(generator, ptr, N, pm);
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

	double expected_avg;
	double expected_var;
	if (pm) {
		expected_avg = 0;
		expected_var = 1. / 3;
	} else {
		expected_avg = 1. / 2;
		expected_var = 1. / 12;
	}

	std::printf("[%15s : %lu : %7s] avg = %+e [expected = %+e], var = %e [expected = %e]\n", get_curand_rng_name_str(rng), N, (pm ? "(-1,1)" : "(0,1)"), avg, expected_avg, var, expected_var);

	mtk::curand_fp16::destroy(generator);
	cudaFree(ptr);
}

void test_curand_fp16_normal(
		const std::size_t N,
		curandRngType_t rng,
		const float mean_in,
		const float var_in
		) {
	half* ptr;
	cudaMallocManaged(&ptr, sizeof(half) * N);

	mtk::curand_fp16::generator_t generator;
	mtk::curand_fp16::create(generator, rng);
	mtk::curand_fp16::set_seed(generator, 0);

	mtk::curand_fp16::normal(generator, ptr, N, mean_in, var_in);
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

	double expected_avg = mean_in;
	double expected_var = var_in;

	std::printf("[%15s] avg = %+e [expected = %+e], var = %e [expected = %e]\n", get_curand_rng_name_str(rng), avg, expected_avg, var, expected_var);

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

void test_throughput_normal(
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
			mtk::curand_fp16::normal(generator, ptr, N, 0, 1);
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
	std::printf("# uniform\n");
	test_curand_fp16(1u << 20, CURAND_RNG_PSEUDO_MRG32K3A     , false);
	test_curand_fp16(1u << 20, CURAND_RNG_PSEUDO_XORWOW       , false);
	test_curand_fp16(1u << 20, CURAND_RNG_PSEUDO_PHILOX4_32_10, false);
	test_curand_fp16(1u << 20, CURAND_RNG_PSEUDO_MRG32K3A     , true );
	test_curand_fp16(1u << 20, CURAND_RNG_PSEUDO_XORWOW       , true );
	test_curand_fp16(1u << 20, CURAND_RNG_PSEUDO_PHILOX4_32_10, true );
	test_curand_fp16((1u << 20) - 1, CURAND_RNG_PSEUDO_MRG32K3A     , false);
	test_curand_fp16((1u << 20) - 1, CURAND_RNG_PSEUDO_XORWOW       , false);
	test_curand_fp16((1u << 20) - 1, CURAND_RNG_PSEUDO_PHILOX4_32_10, false);
	test_curand_fp16((1u << 20) - 1, CURAND_RNG_PSEUDO_MRG32K3A     , true );
	test_curand_fp16((1u << 20) - 1, CURAND_RNG_PSEUDO_XORWOW       , true );
	test_curand_fp16((1u << 20) - 1, CURAND_RNG_PSEUDO_PHILOX4_32_10, true );

	std::printf("# normal\n");
	test_curand_fp16_normal(1u << 20, CURAND_RNG_PSEUDO_MRG32K3A     , 0, 1);
	test_curand_fp16_normal(1u << 20, CURAND_RNG_PSEUDO_XORWOW       , 0, 1);
	test_curand_fp16_normal(1u << 20, CURAND_RNG_PSEUDO_PHILOX4_32_10, 0, 1);
	test_curand_fp16_normal(1u << 20, CURAND_RNG_PSEUDO_MRG32K3A     , 1, 2);
	test_curand_fp16_normal(1u << 20, CURAND_RNG_PSEUDO_XORWOW       , 1, 2);
	test_curand_fp16_normal(1u << 20, CURAND_RNG_PSEUDO_PHILOX4_32_10, 1, 2);
	test_curand_fp16_normal((1u << 20) - 1, CURAND_RNG_PSEUDO_MRG32K3A     , 0, 1);
	test_curand_fp16_normal((1u << 20) - 1, CURAND_RNG_PSEUDO_XORWOW       , 0, 1);
	test_curand_fp16_normal((1u << 20) - 1, CURAND_RNG_PSEUDO_PHILOX4_32_10, 0, 1);
	test_curand_fp16_normal((1u << 20) - 1, CURAND_RNG_PSEUDO_MRG32K3A     , 1, 2);
	test_curand_fp16_normal((1u << 20) - 1, CURAND_RNG_PSEUDO_XORWOW       , 1, 2);
	test_curand_fp16_normal((1u << 20) - 1, CURAND_RNG_PSEUDO_PHILOX4_32_10, 1, 2);

	test_throughput(CURAND_RNG_PSEUDO_MRG32K3A     );
	test_throughput(CURAND_RNG_PSEUDO_XORWOW       );
	test_throughput(CURAND_RNG_PSEUDO_PHILOX4_32_10);
	test_throughput_normal(CURAND_RNG_PSEUDO_MRG32K3A     );
	test_throughput_normal(CURAND_RNG_PSEUDO_XORWOW       );
	test_throughput_normal(CURAND_RNG_PSEUDO_PHILOX4_32_10);
}
