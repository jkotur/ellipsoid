#ifndef __IMAGECONV_H__

#define __IMAGECONV_H__

#include <string>
#include <gtkmm.h>

#include "GlDrawing.h"
#include "RayCasting.h"

class Application {
public:
	Application( const std::string& ui_file );
	virtual ~Application();

	void show()
	{	win_main->show_all(); }

	void quit()
	{	Gtk::Main::quit(); }
protected:
	RayCasting renderer;

	GlDrawingArea*glArea;

	Gtk::Window	*win_main;
	Gtk::Statusbar	*statbar;

	Gtk::SpinButton	*sp_width;
	Gtk::SpinButton	*sp_height;
	Gtk::SpinButton	*sp_step;

	Glib::RefPtr<Gtk::ListStore> ls_time;

	Glib::RefPtr<Gtk::Builder> refBuilder;

	sigc::connection anim;
};

#endif /* __IMAGECONV_H__ */

