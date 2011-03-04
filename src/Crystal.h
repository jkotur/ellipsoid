#ifndef __CRYSTAL_H__

#define __CRYSTAL_H__

#include "buffer.h"

class Crystal {
public:
	Crystal ();
	virtual ~Crystal();

	void init( int width , int height );
	float step();
	void updateGl();

	BufferGl*getAtomsBuffer()
	{	return &glabuf; }
	int getAtomsBufferSize() 
	{	return num; }

	BufferGl*getCrystalBuffer()
	{	return &glcbuf; }
	int getCrystalBufferSize()
	{	return width*height; }
	
private:

	int width , height , num ;

	BufferGl glabuf;
	BufferGl glcbuf;

	BufferCu<short2> atoms;
	BufferCu<char> crystal;
	BufferCu<float>  rnd;
};

#endif /* __CRYSTAL_H__ */

