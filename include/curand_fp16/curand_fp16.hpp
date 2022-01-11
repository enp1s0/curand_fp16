#ifndef __CURAND_FP16__
#define __CURAND_FP16__
#include <cstdint>
#include <curand_kernel.h>
#include <curand.h>
#include <cuda_fp16.h>

namespace mtk {
namespace curand_fp16 {
template <curandRngType_t rng>
struct curand_status_t {using type = void;};
template <> struct curand_status_t<CURAND_RNG_PSEUDO_MTGP32          > {using type = curandStateMtgp32_t;};
template <> struct curand_status_t<CURAND_RNG_QUASI_SCRAMBLED_SOBOL32> {using type = curandStateScrambledSobol32_t;};
template <> struct curand_status_t<CURAND_RNG_QUASI_SOBOL32          > {using type = curandStateSobol32_t;};
template <> struct curand_status_t<CURAND_RNG_PSEUDO_MRG32K3A        > {using type = curandStateMRG32k3a_t;};
template <> struct curand_status_t<CURAND_RNG_PSEUDO_XORWOW          > {using type = curandStateXORWOW_t;};
template <> struct curand_status_t<CURAND_RNG_PSEUDO_PHILOX4_32_10   > {using type = curandStatePhilox4_32_10_t;};

struct generator_t {
	unsigned num_sm;
	unsigned num_threads;
	cudaStream_t cuda_stream;
	curandRngType_t rng_type;
	void* status_ptr;
};

void create(generator_t& gen, const curandRngType_t rng_type);
void destroy(generator_t& gen);
void set_seed(generator_t& gen, const std::uint64_t seed);

// Uniform rand distribution
// pm == true  | (-1, 1)
//       false | ( 0, 1)
void uniform(generator_t& gen, half* const ptr, const std::size_t size, const bool pm = false);
} // namespace curand_fp16
} // namespace mtk
#endif // __CUDA_RAND_FP16__
