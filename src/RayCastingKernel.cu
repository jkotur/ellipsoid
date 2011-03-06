#ifndef __RAYCASTINGKERNEL_CU__

#define __RAYCASTINGKERNEL_CU__

#include <GL/glew.h>

#include <vector_types.h>

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

__global__ void render_elipsoid( GLubyte*screen , uint qw , uint qh , uint w , uint h , uint n , float m )
{
	uint ix = qw * blockIdx.x + threadIdx.x;
	uint iy = qh * blockIdx.y;

	float A = .25;
	float B = .75;
	float C = .5;

	if( ix >= w ) return;

	float3 p  = make_float3( ((float)(blockIdx.x*qw)/(float)w - .5f)*2.0f ,
				 ((float)(blockIdx.y*qh)/(float)h - .5f)*2.0f ,
				 0.0f );

	float3 v  = make_float3( 0.0f , 0.0f , 1.0f );

	float AA = A*A;
	float BB = B*B;
	float CC = C*C;

	float a = -v.x*v.x*BB*CC - v.y*v.y*AA*CC - v.z*v.z*AA*BB;
	float b = -2.0f*( v.x*BB*CC*p.x + v.y*AA*CC*p.y + v.z*AA*BB*p.z );
	float c = AA*BB*CC - AA*BB*p.z*p.z - AA*CC*p.y*p.y - BB*CC*p.x*p.x;

	float d = b*b - 4*a*c;

	uchar3 color;

	if( d >= 0 ) {
		d = sqrt(d);
		float t = min( (-b+d)/(2.0f*a) , (-b-d)/(2.0f*a) );

		float3 q = make_float3( v.x*t + p.x , v.y*t + p.y , v.z*t + p.z );
		float3 n = make_float3( (2.0f/AA)*q.x , 2.0f/BB*q.y , (2.0f/CC)*q.z );
		n /= len(n);

/*                float i = pow( dot( n , v ) , m );*/
		float i =  pow( abs( dot( n , v ) ) , m );

		color = make_uchar3( 200.0f*i+55 , 200.0f*i+55 , 200.0f*i+55 );
	} else	color = make_uchar3( 0 , 0 , 0 );


	uint idx;
	for( uint i = 0 ; i<qh ; i++ )
	{
		if( iy+i >= h ) return;

		idx = (ix + (iy+i)*w)*3;

		screen[idx  ] = color.x;
		screen[idx+1] = color.y;
		screen[idx+2] = color.z;
	}
}

#endif /* __RAYCASTINGKERNEL_CU__ */

