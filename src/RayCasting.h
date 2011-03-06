#ifndef __CRYSTAL_H__

#define __CRYSTAL_H__

#include "buffer.h"

class RayCasting {
public:
	RayCasting();
	virtual ~RayCasting();

	void resize( int width , int height );
	bool render_frame();
	void updateGl();

	BufferGl*getPixBuff()
	{	return &pbo; }
	
private:
	int width , height , num ;
	int step;

	BufferGl pbo;
};

#endif /* __CRYSTAL_H__ */

