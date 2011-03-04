#ifndef __GLDRAWING_H__

#define __GLDRAWING_H__

#include <GL/glew.h>
#include <GL/gl.h>

#include <gtkmm.h>
#include <gtkglmm.h>

#include "RayCasting.h"
#include "buffer.h"

class GlDrawingArea : public Gtk::DrawingArea ,
                      public Gtk::GL::Widget<GlDrawingArea>
{
public:
	GlDrawingArea(BaseObjectType* cobject, const Glib::RefPtr<Gtk::Builder>& builder);
	virtual ~GlDrawingArea();

	void setRenderer( RayCasting*rnd )
	{
		renderer = rnd;
	}

	void setPixBuff( BufferGl*_buf )
	{	pbo = _buf; }

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

	RayCasting*renderer;
	BufferGl*pbo;
	Glib::RefPtr<Gtk::Builder> refBuilder;
};

#endif /* __GLDRAWING_H__ */

