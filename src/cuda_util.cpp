#include "cuda_util.h"

#include <cuda_runtime_api.h>

#include "logger.h"

void CUT_DEVICE_QUERY()
{
	log_printf(INFO,"CUDA Device Query (Runtime API) version (CUDART static linking)\n");

	int deviceCount;

	cudaGetDeviceCount(&deviceCount);

	// This function call returns 0 if there are no CUDA capable devices.
	if (deviceCount == 0)
		log_printf(INFO,"There is no device supporting CUDA\n");
	int dev;
	for (dev = 0; dev < deviceCount; ++dev) {
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, dev);

		if (dev == 0) {
			// This function call returns 9999 for both major & minor fields, if no CUDA capable devices are present
			if (deviceProp.major == 9999 && deviceProp.minor == 9999)
				log_printf(INFO,"There is no device supporting CUDA.\n");
			else if (deviceCount == 1)
				log_printf(INFO,"There is 1 device supporting CUDA\n");
			else
				log_printf(INFO,"There are %d devices supporting CUDA\n", deviceCount);
		}
		log_printf(INFO,"\n");
		log_printf(INFO,"Device %d: \"%s\"\n", dev, deviceProp.name);
#if CUDART_VERSION >= 2020
		int driverVersion = 0, runtimeVersion = 0;
		cudaDriverGetVersion(&driverVersion);
		log_printf(INFO,"  CUDA Driver Version:                           %d.%d\n", driverVersion/1000, driverVersion%100);
		cudaRuntimeGetVersion(&runtimeVersion);
		log_printf(INFO,"  CUDA Runtime Version:                          %d.%d\n", runtimeVersion/1000, runtimeVersion%100);
#endif

		log_printf(INFO,"  CUDA Capability Major revision number:         %d\n", deviceProp.major);
		log_printf(INFO,"  CUDA Capability Minor revision number:         %d\n", deviceProp.minor);

		log_printf(INFO,"  Total amount of global memory:                 %u bytes\n", deviceProp.totalGlobalMem);
#if CUDART_VERSION >= 2000
		log_printf(INFO,"  Number of multiprocessors:                     %d\n", deviceProp.multiProcessorCount);
		log_printf(INFO,"  Number of cores:                               %d\n", 8 * deviceProp.multiProcessorCount);
#endif
		log_printf(INFO,"  Total amount of constant memory:               %u bytes\n", deviceProp.totalConstMem); 
		log_printf(INFO,"  Total amount of shared memory per block:       %u bytes\n", deviceProp.sharedMemPerBlock);
		log_printf(INFO,"  Total number of registers available per block: %d\n", deviceProp.regsPerBlock);
		log_printf(INFO,"  Warp size:                                     %d\n", deviceProp.warpSize);
		log_printf(INFO,"  Maximum number of threads per block:           %d\n", deviceProp.maxThreadsPerBlock);
		log_printf(INFO,"  Maximum sizes of each dimension of a block:    %d x %d x %d\n",
				deviceProp.maxThreadsDim[0],
				deviceProp.maxThreadsDim[1],
				deviceProp.maxThreadsDim[2]);
		log_printf(INFO,"  Maximum sizes of each dimension of a grid:     %d x %d x %d\n",
				deviceProp.maxGridSize[0],
				deviceProp.maxGridSize[1],
				deviceProp.maxGridSize[2]);
		log_printf(INFO,"  Maximum memory pitch:                          %u bytes\n", deviceProp.memPitch);
		log_printf(INFO,"  Texture alignment:                             %u bytes\n", deviceProp.textureAlignment);
		log_printf(INFO,"  Clock rate:                                    %.2f GHz\n", deviceProp.clockRate * 1e-6f);
#if CUDART_VERSION >= 2000
		log_printf(INFO,"  Concurrent copy and execution:                 %s\n", deviceProp.deviceOverlap ? "Yes" : "No");
#endif
#if CUDART_VERSION >= 2020
		log_printf(INFO,"  Run time limit on kernels:                     %s\n", deviceProp.kernelExecTimeoutEnabled ? "Yes" : "No");
		log_printf(INFO,"  Integrated:                                    %s\n", deviceProp.integrated ? "Yes" : "No");
		log_printf(INFO,"  Support host page-locked memory mapping:       %s\n", deviceProp.canMapHostMemory ? "Yes" : "No");
		log_printf(INFO,"  Compute mode:                                  %s\n", deviceProp.computeMode == cudaComputeModeDefault ?
				"Default (multiple host threads can use this device simultaneously)" :
				deviceProp.computeMode == cudaComputeModeExclusive ?
				"Exclusive (only one host thread at a time can use this device)" :
				deviceProp.computeMode == cudaComputeModeProhibited ?
				"Prohibited (no host thread can use this device)" :
				"Unknown");
#endif
	}
}

