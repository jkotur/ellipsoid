#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <GL/glew.h>
#include <GL/gl.h>

#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

#include "RayCasting.h"
#include "RayCastingKernel.cu"

#include "logger.h"


RayCasting::RayCasting( float a , float b , float c , float _m )
	: width(0) , height(0) , step(0) , d_m(NULL)
{
	e.a = a;
	e.b = b;
	e.c = c;
	e.m =_m;

	float i[16] = { 1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1 };

	memcpy( m , i , sizeof(float)*16 );
}

RayCasting::~RayCasting()
{
	if( d_m ) {
		cudaFree( d_m );
		CUT_CHECK_ERROR("RayCasting::~RayCasting::cudaFree");
	}
}

void RayCasting::translate( float x , float y , float z )
{
	float t[16] = { 1, 0, 0, x,
			0, 1, 0, y,
			0, 0, 1, z,
			0, 0, 0, 1 };

	matmul4( m  , t , m);
}

void RayCasting::scale( float x , float y , float z )
{
	float s[16] = { x, 0, 0, 0,
			0, y, 0, 0,
			0, 0, z, 0,
			0, 0, 0, 1 };

	matmul4( m , s , m );
}

void RayCasting::rotate( float a , float x , float y , float z )
{
	float c = cos( a );
	float s = sin( a );
	float xx = x*x;
	float yy = y*y;
	float zz = z*z;

	float r[16] = {  xx+(1-xx)*c  , x*y*(1-c)-z*s , x*z*(1-c)+y*s , 0 ,
			x*y*(1-c)+z*s ,  yy+(1-yy)*c  , y*z*(1-c)-x*s , 0 ,
			x*z*(1-c)-y*s , y*z*(1-c)+x*s ,  zz+(1-zz)*c  , 0 ,
			      0       ,       0       ,       0       , 1 };

	matmul4( m , r , m );
}

void RayCasting::resize( int w , int h )
{
	if( !d_m ) { 
		cudaMalloc( (void**)&d_m , sizeof(float)*16 );
		CUT_CHECK_ERROR("RayCasting::RayCasting::cudaMalloc");
	}

	width = w; height = h;

	GLubyte*d_ub;
	cudaGLMapBufferObject( (void**)&d_ub , pbo.pbo );
	CUT_CHECK_ERROR("RayCasting::init::cudaGLMapBufferObject");

	cudaMemset( d_ub , 0 , sizeof(GLubyte)*w*h*3 );
	CUT_CHECK_ERROR("RayCasting::init::cudaMemset");

	cudaGLUnmapBufferObject( pbo.pbo );
	CUT_CHECK_ERROR("RayCasting::init::cudaGLUnmapBufferObject");
}

bool RayCasting::render_frame( bool next )
{
	unsigned int quads = pow(2,step);

	if( next && (quads < width || quads < height) )
		++step;

	dim3 threads;
	while( (threads = std::ceil( (float)width / (float)(quads=pow(2,step)))).x >= 512 ) ++step;

	unsigned int qw = quads , qh = quads;
	while( qw > width ) qw >>= 1;
	while( qh > height) qh >>= 1;
	qw <<= 1 ; qh <<= 1;
	dim3 blocks  = dim3( quads , qh );

	GLubyte*d_ub;

	cudaGLMapBufferObject( (void**)&d_ub , pbo.pbo );
	CUT_CHECK_ERROR("RayCasting::init::cudaGLMapBufferObject");

	log_printf(DBG,"width %d\theight %d\n",width,height);
	log_printf(DBG,"thr: %d\tblk: %d %d\n",threads.x,blocks.x,blocks.y);

/*        for( int i=0 ; i<16 ; i++ ) printf("%f%c",m[i],i%4-3?' ':'\n');*/
/*        printf("\n");*/

	cudaMemcpy( (void**)d_m , (void**)m , sizeof(float)*16 , cudaMemcpyHostToDevice );
	CUT_CHECK_ERROR("RayCasting::render_frame::cudaMemcpy");

	cudaMemset( d_ub , 0 , sizeof(GLubyte)*width*height*3 );
	CUT_CHECK_ERROR("RayCasting::render_frame::cudaMemset");

	render_elipsoid<<< blocks , threads >>>( d_ub , std::ceil( (float)width / (float)quads  ) , std::ceil( (float)height / (float)quads  ), width , height , e , d_m );
	CUT_CHECK_ERROR("RayCasting::render_frame::render_elipsoid");

	cudaGLUnmapBufferObject( pbo.pbo );
	CUT_CHECK_ERROR("RayCasting::render_frame::cudaGLUnmapBufferObject");

	if( quads < width || quads < height )
		return false;

	return true;
}

