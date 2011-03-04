#include <cmath>
#include <cstdio>
#include <algorithm>

#include <GL/glew.h>
#include <GL/gl.h>

#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

#include "Crystal.h"
#include "CrystalKernel.cu"

#include "constants.h"

#include "rng/rng.h"

using std::printf;

Crystal::Crystal()
{
}

Crystal::~Crystal()
{
}

void Crystal::init( int w , int h )
{
	width  = w;
	height = h;
	num    = std::ceil( (float)(w*h)*MAX_PT ) ;

	printf("%d %d %f\n",w,h,(float)(w*h)*MAX_PT);
/*        num = 100;*/

	atoms  .resize( num );
	crystal.resize( width * height );

	cudaMemset( atoms.d_ptr , -1 , sizeof(short2)*num);
	CUT_CHECK_ERROR( "Crystal::init: memset 1" );

	cudaMemset( crystal.d_ptr , 0 , sizeof(char)*w*h);
	CUT_CHECK_ERROR( "Crystal::init: memset 2" );

/*        defect_init<<< 1 , w/2 >>>( crystal.d_ptr+(h/2)*w , 79 , 79 , w/2 );*/
	cudaMemset( crystal.d_ptr+(h/2)*w     , 79 , w/2 );
	cudaMemset( crystal.d_ptr+(h/2)*w     , 95 , 1   );
	cudaMemset( crystal.d_ptr+(h/2)*w+w/2 , 69 , 1   );
	CUT_CHECK_ERROR( "Crystal::init: kernel launch 1" );

/*        defect_init<<< 1 , w/2 >>>( crystal.d_ptr+(h/2-1)*w , 3 , 3 , w/2 );*/
	cudaMemset( crystal.d_ptr+(h/2-1)*w     , 3 , w/2 );
	cudaMemset( crystal.d_ptr+(h/2-1)*w     , 7 , 1   );
	cudaMemset( crystal.d_ptr+(h/2-1)*w+w/2 , 5 , 1   );
	CUT_CHECK_ERROR( "Crystal::init: kernel launch 2" );

	rnd.resize( num );
	rngStartDevC_h( rnd.d_ptr , num );
	CUT_CHECK_ERROR( "Crystal::init: init random" );
}

float Crystal::step()
{
/*        int v[5084];*/
/*        cudaMemcpy(v,atoms.d_ptr,sizeof(int)*num*2,cudaMemcpyDeviceToHost);*/
/*        for( int i=0 ;i<num*2;i+=2 )*/
/*                printf("(%d %d) ",v[i],v[i+1]);*/
/*        printf("\n");*/

/*        float vf[5084];*/
/*        cudaMemcpy(vf,rnd.d_ptr,sizeof(float)*num,cudaMemcpyDeviceToHost);*/
/*        for( int i=0 ;i<num;i++ )*/
/*                printf("%f ",vf[i]);*/
/*        printf("\n");*/

/*        static int i = 0 ;*/
/*        i++;*/
/*        if( i == (width > height ? width : height) ) {*/
/*                i = 0;*/
/*                cudaMemset( atoms.d_ptr , 0 , sizeof(int)*num*2);*/
/*                CUT_CHECK_ERROR( "Crystal::init: memset 1" );*/
/*        }*/

/*        atoms_rand<<< 1 , num >>>( atoms.d_ptr , num , crystal.d_ptr , width , height , rnd.d_ptr );*/
/*        CUT_CHECK_ERROR( "Crystal::step: rand kernel launch" );*/

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	dim3 threads = std::min( num , 512 );
	dim3 blocks  = std::ceil( num / threads.x );

	cudaEventRecord(start, 0);
	atoms_move<<< blocks , threads >>>( atoms.d_ptr , num , crystal.d_ptr , width , height , rnd.d_ptr );
	cudaEventRecord(stop, 0);
	CUT_CHECK_ERROR( "Crystal::step: move kernel launch" );

	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

/*        char c[5084];*/
/*        cudaMemcpy(c,crystal.d_ptr,sizeof(char)*width*height,cudaMemcpyDeviceToHost);*/
/*        for( int y=height-1 ; y>=0 ; y-- ) {*/
/*                for( int x=0 ; x<width ; x++ )*/
/*                        printf("%3d ",c[x+y*width]);*/
/*                printf("\n");*/
/*        }*/
/*        cudaMemcpy(v,atoms.d_ptr,sizeof(int)*num*2,cudaMemcpyDeviceToHost);*/
/*        for( int i=0 ;i<num*2;i+=2 )*/
/*                printf("(%d %d) ",v[i],v[i+1]);*/
/*        printf("\n");*/

/*        printf("\n");*/

	return elapsedTime;
}

void Crystal::updateGl()
{
	float2*d_g;
	cudaGLMapBufferObject( (void**)&d_g , glabuf.vbo );
	CUT_CHECK_ERROR( "Crystal::updateGl: map buffer" );

	dim3 at_threads = std::min( num , 512 );
	dim3 at_blocks  = std::ceil( num / at_threads.x );

	atoms_to_gl<<< at_blocks , at_threads >>>( d_g , atoms.d_ptr , num );
	CUT_CHECK_ERROR( "Crystal::updateGl: kernel launch" );

	cudaGLUnmapBufferObject( glabuf.vbo );
	CUT_CHECK_ERROR( "Crystal::updateGl: unmap buffer" );


	cudaGLMapBufferObject( (void**)&d_g , glcbuf.vbo );
	CUT_CHECK_ERROR( "Crystal::updateGl: map buffer" );

	cudaMemset( d_g , 0 , sizeof(float)*width*height*2 );
	CUT_CHECK_ERROR( "Crystal::init: memset" );

	dim3 cy_threads = std::min( height , 512 );
	dim3 cy_blocks  = std::ceil( height / cy_threads.x );

	crystal_to_gl<<< cy_blocks , cy_threads >>>( d_g , crystal.d_ptr , width , height );
	CUT_CHECK_ERROR( "Crystal::updateGl: kernel launch" );

	cudaGLUnmapBufferObject( glcbuf.vbo );
	CUT_CHECK_ERROR( "Crystal::updateGl: unmap buffer" );
}

