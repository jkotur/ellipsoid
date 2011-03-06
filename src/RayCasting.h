#ifndef __CRYSTAL_H__

#define __CRYSTAL_H__

#include "buffer.h"

class RayCasting {
public:
	RayCasting( float a , float b , float c , float m );
	virtual ~RayCasting();

	void resize( int width , int height );
	bool render_frame();
	void updateGl();

	BufferGl*getPixBuff()
	{	return &pbo; }

	void set_m( float _m )
	{	m = _m; step=1; }
	
private:
	int width , height , num ;
	int step;

	float a , b , c , m;

	BufferGl pbo;
};

#endif /* __CRYSTAL_H__ */

