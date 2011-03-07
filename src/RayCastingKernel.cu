#ifndef __RAYCASTINGKERNEL_CU__

#define __RAYCASTINGKERNEL_CU__

#include <GL/glew.h>

#include <vector_types.h>

#include "RayCasting.h"

#include "constants.h"

typedef unsigned int uint;

__device__ float len( const float3& v )
{
	return sqrt(v.x*v.x+v.y*v.y+v.z*v.z);
}

__device__ void operator/=( float3& v , float s )
{
	v.x /= s;
	v.y /= s;
	v.z /= s;
}

__device__ float dot( const float3& a , const float3& b )
{
	return a.x*b.x + a.y*b.y + a.z*b.z;
}

#define A(row, col) a[(col << 2) + row]
#define B(row, col) b[(col << 2) + row]

__device__ void
matmulvec4( float          *product,
            const float    *b , /**< vector */
            const float    *a ) /**< matrix */
{
	float vec[4];
	int i;
	for (i = 0; i < 4; i++)
		vec[i] = A(0,i) * B(0,0) + A(1,i) * B(1,0) +
			 A(2,i) * B(2,0) + A(3,i) * B(3,0);
	memcpy(product,vec,sizeof(float)*4);
}

__global__ void render_elipsoid( GLubyte*screen , uint qw , uint qh , uint w , uint h , Elipsoid e , float*m )
{
	uint ix = qw * blockIdx.x + threadIdx.x;
	uint iy = qh * blockIdx.y;

	if( ix >= w ) return;

	float3 p  = make_float3( ((float)(blockIdx.x*qw + qw/2.0f)/(float)w - .5f)*2.0f ,
				 ((float)(blockIdx.y*qh + qh/2.0f)/(float)h - .5f)*2.0f*h/w ,
				 0.0f );

	float3 v  = make_float3( 0.0f , 0.0f , 1.0f );

	float tmp[4] = { p.x , p.y , p.z , 1 };
	matmulvec4( tmp , tmp , m );
	p.x = tmp[0]/tmp[3];
	p.y = tmp[1]/tmp[3];
	p.z = tmp[2]/tmp[3];

	tmp[0]=v.x; tmp[1]=v.y; tmp[2]=v.z; tmp[3]=0.0f;
	matmulvec4( tmp , tmp , m );
	v.x = tmp[0];
	v.y = tmp[1];
	v.z = tmp[2];

	float AA = e.a*e.a;
	float BB = e.b*e.b;
	float CC = e.c*e.c;

	float a = -v.x*v.x*BB*CC - v.y*v.y*AA*CC - v.z*v.z*AA*BB;
	float b = -2.0f*( v.x*BB*CC*p.x + v.y*AA*CC*p.y + v.z*AA*BB*p.z );
	float c = AA*BB*CC - AA*BB*p.z*p.z - AA*CC*p.y*p.y - BB*CC*p.x*p.x;

	float d = b*b - 4*a*c;

	if( d < 0 ) return;

	d = sqrt(d);
	float t = min( (-b+d)/(2.0f*a) , (-b-d)/(2.0f*a) );

	float3 q = make_float3( v.x*t + p.x , v.y*t + p.y , v.z*t + p.z );
	float3 n = make_float3( (2.0f/AA)*q.x , 2.0f/BB*q.y , (2.0f/CC)*q.z );
	n /= len(n);

	float i =  pow( abs( dot( n , v ) ) , e.m );

	int color;
	GLubyte*uc = (unsigned char*)&color;
	uc[0] = 200.0f*i+55;
	uc[1] = 200.0f*i+55;
	uc[2] = 200.0f*i+55;

/*        uint idx = (qw * blockIdx.x + (iy+threadIdx.x)*w)*3;*/
/*        memset( screen + idx , uc[0] , sizeof(unsigned char)*qw*3 );*/

	uint idx;
	for( uint i = 0 ; i<qh ; i++ )
	{
		if( iy+i >= h ) return;

		idx = (ix + (iy+i)*w)*3;

		memcpy( screen + idx , uc , sizeof(GLubyte)*3 );
	}
}

#endif /* __RAYCASTINGKERNEL_CU__ */

