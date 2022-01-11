cmake_minimum_required(VERSION 3.18)
project(curand_fp16 LANGUAGES CXX CUDA)

find_package(CUDA 10.0 REQUIRED)
find_package(CUDAToolkit REQUIRED)

# CUDA/CXX
foreach(lang CXX CUDA)
    set(CMAKE_${lang}_STANDARD          17)
    set(CMAKE_${lang}_STANDARD_REQUIRED ON)
    set(CMAKE_${lang}_EXTENSIONS        OFF)
endforeach()

# CUDA (1/2)
SET(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)

# Directories
set(INCDIR include)
set(SRCDIR src)

file(GLOB HEADERS "${INCDIR}/curand_fp16/*.hpp")

SET(lib_name curand_fp16)
add_library(${lib_name} STATIC
	${SRCDIR}/curand_fp16.cu
	${HEADERS}
	)

target_include_directories(${lib_name} PUBLIC ${INCDIR})
target_link_libraries(${lib_name} PRIVATE
	cuda curand
	)

set_target_properties(${lib_name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(${lib_name} PROPERTIES PUBLIC_HEADER ${INCDIR}/curand_fp16/${library}.hpp)

##########################################################################
# Installing
##########################################################################
install(TARGETS curand_fp16
	LIBRARY DESTINATION lib
	PUBLIC_HEADER DESTINATION include/curand_fp16
	)

##########################################################################
# Tests
##########################################################################

# Directory
set(TESTSRCDIR tests)

foreach(test uniform)
	set(test_out ${test}.test)
	add_executable(${test_out} ${TESTSRCDIR}/${test}.cu ${HEADERS})
	target_include_directories(${test_out} PRIVATE ${INCDIR})
	target_link_libraries(${test_out} PRIVATE curand_fp16)
	target_link_libraries(${test_out} PRIVATE
		CUDA::curand
		)
endforeach()
