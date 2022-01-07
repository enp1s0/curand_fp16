#include <curand_fp16/curand_fp16.hpp>
#include <stdexcept>

namespace {
constexpr unsigned block_size = 512;
template <class T>
__global__ void status_init_kernel(
		T* const status_ptr,
		const std::uint64_t seed
		) {
	const auto tid = blockDim.x * blockIdx.x + threadIdx.x;
	curand_init(seed, tid, 0, status_ptr + tid);
}

template <class T>
struct size_of{int value = 0;};
template <> struct size_of<ushort1> {static const int value = 2;};
template <> struct size_of<uint1  > {static const int value = 4;};
template <> struct size_of<ulong2 > {static const int value = 16;};
template <> struct size_of<half   > {static const int value = 2;};
template <> struct size_of<half2  > {static const int value = 4;};

template <class RNG_T>
__global__ void generate_kernel(
		half* const array_ptr,
		RNG_T* const status_ptr,
		const std::size_t size
		) {
	const auto batch_size = size_of<ulong2>::value / size_of<half>::value;
	const auto tid = blockDim.x * blockIdx.x + threadIdx.x;
	auto curand_gen = *(status_ptr + tid);
	for (unsigned i = tid * batch_size; i < size; i += batch_size * gridDim.x * blockDim.x) {
		const auto res = size - i;
		if (res < batch_size) {
			// TODO
		} else {
			// block gen
			union {
				half   h1[size_of<ulong2>::value / size_of<half >::value];
				half2  h2[size_of<ulong2>::value / size_of<half2>::value];
				ulong2 ul2;
			} batch_block;

			for (unsigned j = 0; j < size_of<ulong2>::value / size_of<uint1>::value; j++) {
				union {
					ushort1 us[size_of<uint1>::value / size_of<ushort1>::value];
					uint1 ui1;
				} rand_batch_block;
				rand_batch_block.ui1.x = curand(&curand_gen);
				for (unsigned k = 0; k < size_of<uint1>::value / size_of<half>::value; k++) {
					const auto us = rand_batch_block.us[k];
					const auto v  = __float2half(static_cast<float>(us.x) / static_cast<float>(0xffff));

					batch_block.h1[k + j * size_of<uint1>::value / size_of<half>::value] = v;
				}
			}
			*reinterpret_cast<ulong2*>(array_ptr + i) = batch_block.ul2;
		}
	}
	*(status_ptr + tid) = curand_gen;
}
} // noname namespace

void mtk::curand_fp16::create(generator_t &gen, const curandRngType_t rng_type) {
	// set cuda stream
	gen.cuda_stream = 0;
	// get num sm
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	gen.num_sm = prop.multiProcessorCount;

	// calculate grid_size
	gen.num_threads = gen.num_sm * 16 * block_size;

	// set algo
	gen.rng_type = rng_type;

	// set generator
	unsigned state_struct_size = 0;
	switch (rng_type) {
#define CASE_RNG_TYPE(rng) case rng: state_struct_size = sizeof(typename mtk::curand_fp16::curand_status_t<rng>::type);break
		CASE_RNG_TYPE(CURAND_RNG_PSEUDO_MRG32K3A        );
		CASE_RNG_TYPE(CURAND_RNG_PSEUDO_XORWOW          );
		CASE_RNG_TYPE(CURAND_RNG_PSEUDO_PHILOX4_32_10   );
		default:
			throw std::runtime_error("Unknown pseudo rand algorithm");
#undef CASE_RNG_TYPE
	}
	const auto stat = cudaMalloc(&gen.status_ptr, state_struct_size * gen.num_threads);
	if (stat != cudaSuccess) {
		throw std::runtime_error("[curand_fp16 error] : " + std::string(cudaGetErrorString(stat)) + " @" + __func__);
	}
}

void mtk::curand_fp16::set_seed(generator_t &gen, const std::uint64_t seed) {
	switch (gen.rng_type) {
#define CASE_RNG_TYPE(rng) case rng: status_init_kernel<typename mtk::curand_fp16::curand_status_t<rng>::type>\
		<<<gen.num_threads / block_size, block_size, 0, gen.cuda_stream>>>\
		(reinterpret_cast<typename mtk::curand_fp16::curand_status_t<rng>::type*>(gen.status_ptr), seed);break
		CASE_RNG_TYPE(CURAND_RNG_PSEUDO_MRG32K3A        );
		CASE_RNG_TYPE(CURAND_RNG_PSEUDO_XORWOW          );
		CASE_RNG_TYPE(CURAND_RNG_PSEUDO_PHILOX4_32_10   );
		default:
			throw std::runtime_error("Unknown pseudo rand algorithm");
#undef CASE_RNG_TYPE
	}
}

void mtk::curand_fp16::uniform(generator_t &gen, half *const ptr, const std::size_t size) {
	switch (gen.rng_type) {
#define CASE_RNG_TYPE(rng) case rng: generate_kernel<typename mtk::curand_fp16::curand_status_t<rng>::type>\
		<<<gen.num_threads / block_size, block_size, 0, gen.cuda_stream>>>\
		(ptr, reinterpret_cast<typename mtk::curand_fp16::curand_status_t<rng>::type*>(gen.status_ptr), size);break
		CASE_RNG_TYPE(CURAND_RNG_PSEUDO_MRG32K3A        );
		CASE_RNG_TYPE(CURAND_RNG_PSEUDO_XORWOW          );
		CASE_RNG_TYPE(CURAND_RNG_PSEUDO_PHILOX4_32_10   );
		default:
			throw std::runtime_error("Unknown pseudo rand algorithm");
#undef CASE_RNG_TYPE
	}
}

void mtk::curand_fp16::destroy(generator_t &gen) {
	cudaFree(gen.status_ptr);
}
