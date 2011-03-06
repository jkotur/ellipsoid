#include <cmath>
#include <cstdio>
#include <algorithm>

#include <GL/glew.h>
#include <GL/gl.h>

#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

#include "RayCasting.h"
#include "RayCastingKernel.cu"

#include "logger.h"

#include "constants.h"

RayCasting::RayCasting()
	: width(0) , height(0) , step(1)
{
}

RayCasting::~RayCasting()
{
}

void RayCasting::resize( int w , int h )
{
	width = w; height = h;

	GLubyte*d_ub;
	cudaGLMapBufferObject( (void**)&d_ub , pbo.pbo );
	CUT_CHECK_ERROR("RayCasting::init::cudaGLMapBufferObject");

	cudaMemset( d_ub , 128 , sizeof(GLubyte)*w*h*3 );
	CUT_CHECK_ERROR("RayCasting::init::cudaMemset");

	cudaGLUnmapBufferObject( pbo.pbo );
	CUT_CHECK_ERROR("RayCasting::init::cudaGLUnmapBufferObject");
}

bool RayCasting::render_frame()
{
	unsigned int quads = pow(2,step);
	dim3 threads = std::ceil( (float)width / (float)quads  );
	dim3 blocks  = dim3( quads , quads );

	GLubyte*d_ub;

	cudaGLMapBufferObject( (void**)&d_ub , pbo.pbo );
	CUT_CHECK_ERROR("RayCasting::init::cudaGLMapBufferObject");

	log_printf(DBG,"width %d\theight %d\n",width,height);
	log_printf(DBG,"thr: %d\tblk: %d %d\n",threads.x,blocks.x,blocks.y);

	render_elipsoid<<< blocks , threads >>>( d_ub , std::ceil( (float)width / (float)quads  ) , std::ceil( (float)height / (float)quads  ), width , height  , quads );

	cudaGLUnmapBufferObject( pbo.pbo );
	CUT_CHECK_ERROR("RayCasting::init::cudaGLUnmapBufferObject");

	if( quads < width || quads < height ) {
		step++;
		return false;
	}

	return true;
}

void RayCasting::updateGl()
{
}

