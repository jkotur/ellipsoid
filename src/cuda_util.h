#ifndef __CUDA_UTIL_H__

#define __CUDA_UTIL_H__

#include <stdlib.h>
#include "logger.h"

#ifdef __DEVICE_EMULATION__
#include <stdio.h>
#define EMUPRINT(...) printf(__VA_ARGS__)
#else
#define EMUPRINT(...)
#endif

#define CUT_CHECK_ERROR(errorMessage) do {                                \
	cudaThreadSynchronize();                                          \
	cudaError_t err = cudaGetLastError();                             \
	if( cudaSuccess != err) {                                         \
		log_printf(CRITICAL,                                      \
			"Cuda error: %s in file '%s' in line %i : %s.\n", \
			errorMessage, __FILE__, __LINE__,                 \
			cudaGetErrorString( err ) );                      \
		exit(EXIT_FAILURE);                                       \
	} } while (0)

void CUT_DEVICE_QUERY();

#endif /* __CUDA_UTIL_H__ */

