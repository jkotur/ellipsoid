#include "GlDrawing.h"

#include <GL/glew.h>
#include <GL/gl.h>
#include <GL/glu.h>

#include <gtkmm.h>
#include <glibmm.h>
#include <gtkglmm.h>

#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

#include <iostream>
#include <cstdlib>

#include "GlUtils.h"

#include "logger.h"
#include "constants.h"
#include "cuda_util.h"


GlDrawingArea::GlDrawingArea(BaseObjectType*cobject, const Glib::RefPtr<Gtk::Builder>& builder)
	: Gtk::DrawingArea(cobject),
	  refBuilder(builder),
	  pbo(NULL) , boxw(INIT_BOX) , boxh(INIT_BOX)
{
	Glib::RefPtr<Gdk::GL::Config> glconfig;
	glconfig = Gdk::GL::Config::create(
			Gdk::GL::MODE_RGB    |
			Gdk::GL::MODE_DEPTH  |
			Gdk::GL::MODE_DOUBLE);
	if (!glconfig) {
		log_printf(_ERROR,"Cannot find the double-buffered visual.\nTrying single-buffered visual.\n");
		glconfig = Gdk::GL::Config::create(Gdk::GL::MODE_RGB |Gdk::GL::MODE_DEPTH);
		if (!glconfig) {
			log_printf(CRITICAL,"Cannot find any OpenGL-capable visual.\n");
			exit(1);
		}
	}
	GLConfigUtil::examine_gl_attrib(glconfig);
	set_gl_capability(glconfig);
}

GlDrawingArea::~GlDrawingArea()
{
}

void GlDrawingArea::initGLEW()
{
        GLenum err = glewInit();                       
        if (GLEW_OK != err) {                          
                  log_printf(CRITICAL,"[GLEW] %s\n",glewGetErrorString(err)); 
		  exit(1);
        }                                              
        log_printf(INFO,"[GLEW] Using GLEW %s\n",glewGetString(GLEW_VERSION));
}

void GlDrawingArea::on_realize()
{
//        log_printf(DBG,"Realize\n");
	// We need to call the base on_realize()
	Gtk::DrawingArea::on_realize();

	Glib::RefPtr<Gdk::GL::Window> glwindow = get_gl_window();

	// *** OpenGL BEGIN ***
	if (!glwindow->gl_begin(get_gl_context()))
		return;

	log_printf(DBG,"realize!!!\n");

	initGLEW();

	glwindow->gl_end();
	// *** OpenGL END ***
}

bool GlDrawingArea::on_configure_event(GdkEventConfigure* event)
{
//        log_printf(DBG,"Configure\n");
	Glib::RefPtr<Gdk::GL::Window> glwindow = get_gl_window();

	if (!glwindow->gl_begin(get_gl_context()))
		return false;

	glViewport(0, 0, get_width(), get_height());

	log_printf(DBG,"%p %d %d\n",pbo,get_width(),get_height());

	glwindow->gl_end();

	return true;
}

bool GlDrawingArea::on_expose_event(GdkEventExpose* event)
{
//        log_printf(DBG,"Expose\n");
	Glib::RefPtr<Gdk::GL::Window> glwindow = get_gl_window();

	// *** OpenGL BEGIN ***
	if (!glwindow->gl_begin(get_gl_context()))
		return false;

	scene_init();
	scene_draw();

	// Swap buffers.
	if (glwindow->is_double_buffered())
		glwindow->swap_buffers();
	else
		glFlush();

	glwindow->gl_end();
	// *** OpenGL END ***

	return true;
}

void GlDrawingArea::scene_init()
{
	glViewport(0, 0, get_width(), get_height());

	glClearColor(0.0, 0.0, 0.0, 0.0);
	glClearDepth(1.0);

	glClear(GL_COLOR_BUFFER_BIT);

	if( pbo->len != get_width()*get_height() )
	{
		int w = ceil((float)get_width()/4.0f)*4;
		int h = get_height();

		bufferResize( pbo , w*h );

		renderer->resize( w,h );
	}
}

void GlDrawingArea::scene_draw()
{
	log_printf(DBG,"cudownie\n");

	renderer->render_frame();

	// FIXME: why the fuck with must be multiplication of 4???

	int w = ceil((float)get_width()/4.0f)*4;
	int h = get_height();

	glBindBuffer(GL_PIXEL_UNPACK_BUFFER,pbo->pbo);
	glDrawPixels(w,h,GL_BGR,GL_UNSIGNED_BYTE,NULL);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER,0);
}

BufferGl*GlDrawingArea::bufferResize( BufferGl*buf , size_t len )
{
	if( !buf ) return NULL;
	if( buf->real_len >= len ) {
		buf->len = len;
		return buf;
	}

	Glib::RefPtr<Gdk::GL::Window> glwindow = get_gl_window();

	if (!glwindow->gl_begin(get_gl_context()))
		return false;

	if( buf->len ) {
		cudaGLUnregisterBufferObject( buf->pbo );
		glDeleteBuffers(1,&buf->pbo);
	}

	glGenBuffers(1,&buf->pbo);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER,buf->pbo);
	glBufferData(GL_PIXEL_UNPACK_BUFFER,
			sizeof(GLubyte)*len*3,
			NULL,GL_STREAM_DRAW);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER,0);

	log_printf(DBG,"Created buffer %d with size %d\n",buf->pbo,len);

	cudaGLRegisterBufferObject(buf->pbo);
	CUT_CHECK_ERROR( "GlDrawingArea::bufferResize: register buffer" );

	buf->real_len = buf->len = len;

	glwindow->gl_end();

	return buf;
}

