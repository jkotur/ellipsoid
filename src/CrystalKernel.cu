#ifndef __CRYSTALKERNEL_CU__

#define __CRYSTALKERNEL_CU__

#include <vector_types.h>

#include "constants.h"

#include "rng/rng.cu"

#define NGET(x,i) ((x)& (1<<i))
#define NSET(x,i) ((x)|=(1<<i))
#define FULL(x)   (((x)&127)==127)
#define WALL(x)   ((x)& 64)
#define WALLIT(x) ((x)|=64)
#define EMPTY(x)  ((x)==0)

const int NN  = 6;
const float GLUE_CHANCE = 0.10f;
const float APPEAR_CHANCE = 0.001f;

__global__ void defect_init( char*g_m , char a , char b , int len )
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if( idx >= len ) return;

	if( idx & 1 )
		g_m[idx] = b;
	else	g_m[idx] = a;
}

/*__device__ int rand_d( int*prev , int mod )*/
/*{*/
/*        return (*prev=rnd_d( *prev , 16811 ))%MOD;*/
/*}*/


__device__ float rand_d( float prev )
{
	return (float)rnd_d( prev*MOD , 16811 )/(float)MOD;
}

__global__ void atoms_rand( int2*g_a , int len  , char*g_c , int w , int h , float*g_rnd )
{
}

#define SX_ODD  580  // 010010000100
#define SY_ODD  90   // 000001011010
#define SX_EVEN 2441 // 100110001001
#define SY_EVEN 90   // 000001011010

#define GETNEB( n , i ) (((n>>(i<<1)) & 3)-1)

__global__ void atoms_move( short2*g_a , int len  , char*g_c , int w , int h , float*g_rnd )
{
/*        char NBX_ODD[] = {-1 , 0 , -1 , 1 ,-1 , 0 };*/ // 580
/*        char NBY_ODD[] = { 1 , 1 ,  0 , 0 ,-1 ,-1 };*/ // 90
/*        char NBX_EVEN[] = { 0 , 1 , -1 , 1 , 0 , 1 };*/ // 2441
/*        char NBY_EVEN[] = { 1 , 1 ,  0 , 0 ,-1 ,-1 };*/ // 90

/*        __shared__ char NBX_ODD[6];// = {-1 , 0 , -1 , 1 ,-1 , 0 };*/
/*        __shared__ char NBY_ODD[6];// = { 1 , 1 ,  0 , 0 ,-1 ,-1 };*/
/*        __shared__ char NBX_EVEN[6];// = { 0 , 1 , -1 , 1 , 0 , 1 };*/
/*        __shared__ char NBY_EVEN[6];// = { 1 , 1 ,  0 , 0 ,-1 ,-1 };*/

/*        if( threadIdx.x == 0 ) NBX_ODD[0] = -1;*/
/*        if( threadIdx.x == 1 ) NBX_ODD[1] =  0;*/
/*        if( threadIdx.x == 2 ) NBX_ODD[2] = -1;*/
/*        if( threadIdx.x == 3 ) NBX_ODD[3] =  1;*/
/*        if( threadIdx.x == 4 ) NBX_ODD[4] = -1;*/
/*        if( threadIdx.x == 5 ) NBX_ODD[5] =  0;*/

/*        if( threadIdx.x == 6 ) NBY_ODD[0] =  1;*/
/*        if( threadIdx.x == 7 ) NBY_ODD[1] =  1;*/
/*        if( threadIdx.x == 8 ) NBY_ODD[2] =  0;*/
/*        if( threadIdx.x == 9 ) NBY_ODD[3] =  0;*/
/*        if( threadIdx.x == 10) NBY_ODD[4] = -1;*/
/*        if( threadIdx.x == 11) NBY_ODD[5] = -1;*/

/*        if( threadIdx.x == 12) NBX_EVEN[0] =  0;*/
/*        if( threadIdx.x == 13) NBX_EVEN[1] =  1;*/
/*        if( threadIdx.x == 14) NBX_EVEN[2] = -1;*/
/*        if( threadIdx.x == 15) NBX_EVEN[3] =  1;*/
/*        if( threadIdx.x == 16) NBX_EVEN[4] =  0;*/
/*        if( threadIdx.x == 17) NBX_EVEN[5] =  1;*/

/*        if( threadIdx.x == 18) NBY_EVEN[0] =  1;*/
/*        if( threadIdx.x == 19) NBY_EVEN[1] =  1;*/
/*        if( threadIdx.x == 20) NBY_EVEN[2] =  0;*/
/*        if( threadIdx.x == 21) NBY_EVEN[3] =  0;*/
/*        if( threadIdx.x == 22) NBY_EVEN[4] = -1;*/
/*        if( threadIdx.x == 23) NBY_EVEN[5] = -1;*/

/*        if( threadIdx.x == 0 ) {*/
/*                NBX_ODD[0] = -1;*/
/*                NBX_ODD[1] =  0;*/
/*                NBX_ODD[2] = -1;*/
/*                NBX_ODD[3] =  1;*/
/*                NBX_ODD[4] = -1;*/
/*                NBX_ODD[5] =  0;*/

/*                NBY_ODD[0] =  1;*/
/*                NBY_ODD[1] =  1;*/
/*                NBY_ODD[2] =  0;*/
/*                NBY_ODD[3] =  0;*/
/*                NBY_ODD[4] = -1;*/
/*                NBY_ODD[5] = -1;*/

/*                NBX_EVEN[0] =  0;*/
/*                NBX_EVEN[1] =  1;*/
/*                NBX_EVEN[2] = -1;*/
/*                NBX_EVEN[3] =  1;*/
/*                NBX_EVEN[4] =  0;*/
/*                NBX_EVEN[5] =  1;*/

/*                NBY_EVEN[0] =  1;*/
/*                NBY_EVEN[1] =  1;*/
/*                NBY_EVEN[2] =  0;*/
/*                NBY_EVEN[3] =  0;*/
/*                NBY_EVEN[4] = -1;*/
/*                NBY_EVEN[5] = -1;*/
/*        }*/

/*        __syncthreads();*/

	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if( idx >= len ) return;

/*        short2 ga = */

	short x = g_a[idx].x;
	short y = g_a[idx].y;

/*        short x = g_a[idx].x;*/
/*        short y = g_a[idx].y;*/

	float nxt = g_rnd[idx];

	short NBX;
	short NBY;

	if( ( x < 0 || y < 0 )
/*          ){*/
	 && (nxt=rand_d(nxt)) < APPEAR_CHANCE ) {
		int nx = (int)((nxt=rand_d(nxt))*w);
		int ny = (int)((nxt=rand_d(nxt))*h);

/*                if( !WALL(g_c[nx+ny*w]) ) {*/
			x = nx;
			y = ny;
/*                }*/
	}

	if( x < 0 || y < 0 || x >= w || y >= h ) {
		g_rnd[idx] = nxt;
		return;
	}

	if( y & 1 ) {
		NBX = SX_ODD;
		NBY = SY_ODD;
	} else { 
		NBX = SX_EVEN;
		NBY = SY_EVEN;
	}

	int rnd = (int)((nxt=rand_d(nxt)) * NN);
/*        rnd %= NN;*/

	int idc = x+y*w;

	bool oldwall = WALL( g_c[idc] );

	x += GETNEB( NBX , rnd );
	y += GETNEB( NBY , rnd );

	if( x >= 0 && x < w
	 && y >= 0 && y < h ) {
		g_a[ idx ].x = x;
		g_a[ idx ].y = y;
		idc = x+y*w;
	} else {
		g_a[ idx ].x = -1;
		g_a[ idx ].y = -1;
		g_rnd[idx] = nxt;
		return;
	}

/*        const char atm = g_c[idc];*/

	if( y & 1 ) {
		NBX = SX_ODD;
		NBY = SY_ODD;
	} else { 
		NBX = SX_EVEN;
		NBY = SY_EVEN;
	}

	int ix , iy;

	int nebs = 0;
	for( int i=0 ; i<NN ; i ++ )
	{
		ix = x + GETNEB( NBX ,i );
		iy = y + GETNEB( NBY ,i );
		if( ix < 0 || ix >= w
		 || iy < 0 || iy >= h )
			continue;
		char n = g_c[ix+iy*w];
		if( NGET( n , NN-1-i ) ) {
			nebs = 0; // another line is too close
			break;
		}
		if( WALL( n ) )
			++nebs;
	}

	if( nebs > 1 && !WALL(g_c[idc]) && !EMPTY(g_c[idc]) && ( oldwall || (nxt=rand_d(nxt)) < (float)(nebs*nebs)*GLUE_CHANCE ) ) {
		WALLIT( g_c[idc] );
		g_a[ idx ].x = -1;
		g_a[ idx ].y = -1;
		for( int i=0 ; i<NN ; i ++ )
		{
			ix = x + GETNEB( NBX , i );
			iy = y + GETNEB( NBY , i );
			if( ix < 0 || ix >= w
			 || iy < 0 || iy >= h ) {
				NSET(g_c[idc],i); // FIXME: this ...
				continue;
			}
			NSET(g_c[ix+iy*w],NN-1-i);
			if( WALL(g_c[ix+iy*w]) )
				NSET(g_c[idc],i); // ...and this is the same operation
		}
		if( FULL( g_c[idc] ) )
			g_c[idc] = 0;
		for( int i=0 ; i<NN ; i++ )
		{
			ix = x + GETNEB( NBX , i );
			iy = y + GETNEB( NBY , i );
			if( FULL(g_c[ix+iy*w]) )
				g_c[ix+iy*w] = 0;
		}
	}

	g_rnd[idx] = nxt;
}

__global__ void atoms_to_gl( float2*g_gl , short2*g_a , int num )
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if( idx >= num ) return;

	float y = g_a[idx].y;
	float x = g_a[idx].x;

	if( ! ((int)y & 1) ) x += 0.5f;
	y *= WIDTH;

	g_gl[ idx ] = make_float2( x , y );
}

__global__ void crystal_to_gl( float2*g_gl , char*g_c , int w , int h )
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if( idx >= h ) return;

	for( int i = 0 ; i < w ; i++ )
	{
		unsigned int id = idx*w + i;
		if( WALL(g_c[ id ]) ) {
			g_gl[ id ].y = idx * WIDTH;
			if( idx & 1 )
				g_gl[ id ].x = i;
			else	g_gl[ id ].x = i + 0.5f;
		}
	}
}

#endif /* __CRYSTALKERNEL_CU__ */

