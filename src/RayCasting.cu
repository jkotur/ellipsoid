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
	: width(0) , height(0)
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

void RayCasting::render_frame()
{
}

void RayCasting::updateGl()
{
}

