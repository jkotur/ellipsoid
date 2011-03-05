#ifndef __RAYCASTINGKERNEL_CU__

#define __RAYCASTINGKERNEL_CU__

#include <GL/glew.h>

#include <vector_types.h>

#include "constants.h"

typedef unsigned int uint;

__global__ void render_elipsoid( GLubyte*screen , uint qw , uint qh , uint w , uint h , uint n )
{
	uint x = qw * blockIdx.x + threadIdx.x;
	uint y = qh * blockIdx.y;

	if( x >= w ) return;

	uint idx;
	for( uint i = 0 ; i<qh ; i++ )
	{
		if( y+i >= h ) return;

		idx = (x + (y+i)*w)*3;

		screen[idx  ] = 0;
		screen[idx+1] = 255.0*((float)blockIdx.y/blockDim.y);
/*                screen[idx+2] = threadIdx.x % 256;*/
		screen[idx+2] = 255.0*((float)blockIdx.x/blockDim.x);
	}
}

#endif /* __RAYCASTINGKERNEL_CU__ */

