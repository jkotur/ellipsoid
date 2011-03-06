#include "Application.h"

#include <string>

#include <GL/glew.h>
#include <GL/gl.h>

#include <gtkmm.h>
#include <gtkglmm.h>

#include "logger.h"

#include "constants.h"

#include "GlDrawing.h"
#include "GlUtils.h"

Application::Application( const std::string& ui_file )
	: renderer(.33,.75,.5,1.0f)
{
	refBuilder = Gtk::Builder::create_from_file(ui_file);
		

	Gtk::Button*but_quit;
//        Gtk::Button*but_pause;
//        Gtk::Button*but_run;
//        Gtk::Button*but_new;
//        Gtk::Button*but_test;

	refBuilder->get_widget( "win_main"  ,win_main);

	refBuilder->get_widget( "but_quit"  ,but_quit);
//        refBuilder->get_widget( "but_run"   ,but_run );
//        refBuilder->get_widget( "but_pause" ,but_pause);
//        refBuilder->get_widget( "but_new"   ,but_new );
//        refBuilder->get_widget( "but_test"  ,but_test );

//        refBuilder->get_widget( "statbar"   ,statbar);

	refBuilder->get_widget( "sp_a" , sp_a );
	refBuilder->get_widget( "sp_b" , sp_b );
	refBuilder->get_widget( "sp_c" , sp_c );
	refBuilder->get_widget( "sp_m" , sp_m );

	sp_m->signal_value_changed()
		.connect( sigc::mem_fun(*this,&Application::on_m_changed) );

//        Glib::RefPtr<Glib::Object> o = refBuilder->get_object( "timestore" );
//        ls_time = Glib::RefPtr<Gtk::ListStore>::cast_static(o);

//        ls_time->append()->set_value( 1 , 1.337 );
//        ls_time->append()->set_value( 2 , 1.2 );

	but_quit->signal_clicked()
		.connect( sigc::mem_fun(*this,&Application::quit) );
//        win_main->signal_delete_event()
//                .connect( sigc::mem_fun(*this,&Application::quit) );
//        but_run ->signal_clicked()
//                .connect( sigc::mem_fun(*this,&Application::run ) );
//        but_new ->signal_clicked()
//                .connect( sigc::mem_fun(*this,&Application::create ) );
//        but_pause->signal_clicked()
//                .connect( sigc::mem_fun(*this,&Application::pause  ) );
//        but_test->signal_clicked()
//                .connect( sigc::mem_fun(*this,&Application::test  ) );

	glArea = NULL;
	refBuilder->get_widget_derived("drawing_gl",glArea);
	glArea->setPixBuff( renderer.getPixBuff()   ); 
	glArea->setRenderer( &renderer );
}

Application::~Application()
{
}

void Application::on_m_changed()
{
	renderer.set_m( sp_m->get_value() );
	glArea->queue_draw();
}

