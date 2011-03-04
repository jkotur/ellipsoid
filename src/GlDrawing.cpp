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
	  rbuf(NULL) , wbuf(NULL) , boxw(INIT_BOX) , boxh(INIT_BOX)
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

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	double box = boxh*WIDTH > boxw ? boxh*WIDTH : boxw ;
	gluOrtho2D(0,box,0,box);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
}

void GlDrawingArea::scene_draw()
{
	glEnableClientState(GL_VERTEX_ARRAY);

	if( rbuf && rbuf->len > 0 ) {
		glBindBuffer(GL_ARRAY_BUFFER,rbuf->vbo);
		glVertexPointer(2,GL_FLOAT,0,NULL);
		glBindBuffer(GL_ARRAY_BUFFER,0);

		glColor3f(1.0,0.0,0.0);
		glDrawArrays(GL_POINTS,0,rbuf->len);
	}

	if( wbuf && wbuf->len > 0 ) {
		glBindBuffer(GL_ARRAY_BUFFER,wbuf->vbo);
		glVertexPointer(2,GL_FLOAT,0,NULL);
		glBindBuffer(GL_ARRAY_BUFFER,0);

		glColor3f(1.0,1.0,1.0);
		glDrawArrays(GL_POINTS,0,wbuf->len);
	}

	glDisableClientState(GL_VERTEX_ARRAY);
}

BufferGl*GlDrawingArea::bufferResize( BufferGl*buf , size_t len )
{
	if( !buf ) buf = new BufferGl();
	if( buf->real_len >= len ) {
		buf->len = len;
		return buf;
	}

	Glib::RefPtr<Gdk::GL::Window> glwindow = get_gl_window();

	if (!glwindow->gl_begin(get_gl_context()))
		return false;

	if( buf->len ) {
		cudaGLUnregisterBufferObject( buf->vbo );
		glDeleteBuffers(1,&buf->vbo);
	}

	glGenBuffers(1,&buf->vbo);
	glBindBuffer(GL_ARRAY_BUFFER,buf->vbo);
	glBufferData(GL_ARRAY_BUFFER,
			sizeof(float)*len*2.0,
			NULL,GL_STREAM_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER,0);

	log_printf(DBG,"Created buffer %d with size %d\n",buf->vbo,len);

	cudaGLRegisterBufferObject(buf->vbo);
	CUT_CHECK_ERROR( "GlDrawingArea::bufferResize: register buffer" );

	buf->real_len = buf->len = len;

	glwindow->gl_end();

	return buf;
}

