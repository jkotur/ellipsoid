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
	bool on_motion(GdkEventMotion* event);
	bool on_button_press(GdkEventButton* event);

	void on_a_changed();
	void on_b_changed();
	void on_c_changed();
	void on_m_changed();
	void on_dt_changed();

	float base_x , base_y;

	RayCasting renderer;

	GlDrawingArea*glArea;

	Gtk::Window	*win_main;
	Gtk::Statusbar	*statbar;

	Gtk::SpinButton	*sp_a;
	Gtk::SpinButton	*sp_b;
	Gtk::SpinButton	*sp_c;
	Gtk::SpinButton	*sp_m;
	Gtk::SpinButton	*sp_dt;

	Gtk::RadioButton *rbut_xy;
	Gtk::RadioButton *rbut_xz;
	Gtk::RadioButton *rbut_yz;

	Gtk::RadioButton *rbut_trans;
	Gtk::RadioButton *rbut_scale;
	Gtk::RadioButton *rbut_rotate;
	Gtk::RadioButton *rbut_isoscale;

	Glib::RefPtr<Gtk::ListStore> ls_time;

	Glib::RefPtr<Gtk::Builder> refBuilder;

	sigc::connection anim;
};

#endif /* __IMAGECONV_H__ */

