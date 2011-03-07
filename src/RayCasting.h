#ifndef __CRYSTAL_H__

#define __CRYSTAL_H__

#include "buffer.h"

struct Elipsoid
{
	float a , b , c;
	float m;
};

class RayCasting {
public:
	RayCasting( float a , float b , float c , float m );
	virtual ~RayCasting();

	void resize( int width , int height );
	bool render_frame( bool next = true );

	void translate( float x , float y , float z );
	void scale( float x , float y , float z );
	void rotate( float a , float x , float y , float z );

	BufferGl*getPixBuff()
	{	return &pbo; }

	void set_a( float _a )
	{	e.a = _a; reset(); }
	void set_b( float _b )
	{	e.b = _b; reset(); }
	void set_c( float _c )
	{	e.c = _c; reset(); }
	void set_m( float _m )
	{	e.m = _m; reset(); }

	void reset()
	{	step=0; }
	
private:
	int width , height , num ;
	int step;

	float m[16];
	float*d_m;

	Elipsoid e;

	BufferGl pbo;
};

#endif /* __CRYSTAL_H__ */

