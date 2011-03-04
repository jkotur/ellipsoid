#ifndef __GLDRAWING_H__

#define __GLDRAWING_H__

#include <GL/glew.h>
#include <GL/gl.h>

#include <gtkmm.h>
#include <gtkglmm.h>

#include "buffer.h"

class GlDrawingArea : public Gtk::DrawingArea ,
                      public Gtk::GL::Widget<GlDrawingArea>
{
public:
	GlDrawingArea(BaseObjectType* cobject, const Glib::RefPtr<Gtk::Builder>& builder);
	virtual ~GlDrawingArea();

	void setRedBuffer( BufferGl*_buf )
	{	rbuf = _buf; }
	void setWhiteBuffer( BufferGl*_buf )
	{	wbuf = _buf; }
	void setZoom( double w , double h )
	{	boxw = w; boxh = h; }

	BufferGl*bufferResize( BufferGl*buf , size_t len );
protected:
	void initGLEW();

	void on_realize();
	bool on_expose_event(GdkEventExpose* event);
	bool on_configure_event(GdkEventConfigure* event);
private:
	void scene_init();
	void scene_draw();

	double boxw , boxh;

	BufferGl*rbuf;
	BufferGl*wbuf;
	Glib::RefPtr<Gtk::Builder> refBuilder;
};

#endif /* __GLDRAWING_H__ */

