#ifndef __BUFFER_H__

#define __BUFFER_H__

#include <GL/glew.h>
#include <GL/gl.h>

#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

#include "cuda_util.h"

#include "logger.h"

struct BufferGl {
	BufferGl() : pbo(0) , len(0) , real_len(0) {}
	~BufferGl() { if( real_len ) glDeleteBuffers(1,&pbo); }

	GLuint pbo;
	size_t len;
	size_t real_len;
};

template<typename T>
struct BufferCu {
	BufferCu() : d_ptr(NULL) , len(0) {}
	virtual ~BufferCu() { if( d_ptr ) cudaFree(d_ptr); }

	void resize( size_t _len )
	{
		if( len >= _len ) return;

		if( d_ptr ) cudaFree(d_ptr);
		cudaMalloc((void**)&d_ptr,_len*sizeof(T));
		CUT_CHECK_ERROR( "malloc" );
		len = _len;
	}

	T*d_ptr;
	size_t len;
};

#endif /* __BUFFER_H__ */

