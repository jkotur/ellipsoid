#include "Application.h"

#include <string>

#include <GL/glew.h>
#include <GL/gl.h>

#include <gtkmm.h>
#include <gtkglmm.h>

#include "timer/timer.h"
#include "logger.h"

#include "constants.h"

#include "GlDrawing.h"
#include "GlUtils.h"

Application::Application( const std::string& ui_file )
{
	refBuilder = Gtk::Builder::create_from_file(ui_file);

	Gtk::Button*but_quit;
	Gtk::Button*but_pause;
	Gtk::Button*but_run;
	Gtk::Button*but_new;
	Gtk::Button*but_test;

	refBuilder->get_widget("win_main",win_main);

	refBuilder->get_widget( "but_quit"  ,but_quit);
	refBuilder->get_widget( "but_run"   ,but_run );
	refBuilder->get_widget( "but_pause" ,but_pause);
	refBuilder->get_widget( "but_new"   ,but_new );
	refBuilder->get_widget( "but_test"  ,but_test );

	refBuilder->get_widget( "statbar"   ,statbar);

	refBuilder->get_widget( "sp_step"   , sp_step   );
	refBuilder->get_widget( "sp_width"  , sp_width  );
	refBuilder->get_widget( "sp_height" , sp_height );

	Glib::RefPtr<Glib::Object> o = refBuilder->get_object( "timestore" );
	ls_time = Glib::RefPtr<Gtk::ListStore>::cast_static(o);

	ls_time->append()->set_value( 1 , 1.337 );
	ls_time->append()->set_value( 2 , 1.2 );

	but_quit->signal_clicked()
		.connect( sigc::mem_fun(*this,&Application::quit) );
	but_run ->signal_clicked()
		.connect( sigc::mem_fun(*this,&Application::run ) );
	but_new ->signal_clicked()
		.connect( sigc::mem_fun(*this,&Application::create ) );
	but_pause->signal_clicked()
		.connect( sigc::mem_fun(*this,&Application::pause  ) );
	but_test->signal_clicked()
		.connect( sigc::mem_fun(*this,&Application::test  ) );

	glArea = NULL;
	refBuilder->get_widget_derived("drawing_gl",glArea);
	glArea->setWhiteBuffer( crystal.getAtomsBuffer()   ); 
	glArea->setRedBuffer  ( crystal.getCrystalBuffer() );
}

Application::~Application()
{
}

void Application::test()
{
	create();
	int w = sp_width ->get_value_as_int(); 
	int h = sp_height->get_value_as_int();
	int s = sp_step->get_value_as_int();
	float avg = 0.0;
	int times = 20;
	log_printf(INFO,"Test started - w: %d  h: %d\n",w,h);
	for( int i = 0 ; i < times ; i++ )
	{
		float time = 0.0;
		for( int i = sp_step->get_value_as_int() ; i-->0 ; ) 
			time += crystal.step();
		log_printf(INFO,"%d steps performed in %f ms\n",s,time);
		avg += time;
	}
	log_printf(INFO,"average time : %f ms\n",avg/(float)times);
	crystal.updateGl();
	glArea->queue_draw();
}

bool Application::step()
{
//        log_printf(DBG,"Step\n");

	timer.start();
	for( int i = sp_step->get_value_as_int() ; i-->0 ; ) 
		crystal.step();
	crystal.updateGl();
	timer.stop();
	statbar->pop();
	statbar->push( Glib::ustring::compose("Computed in %1 ms",timer.get_ms() ) );
	glArea->queue_draw();
	return true;
}

void Application::run()
{
	if( anim ) return;

	log_printf(DBG,"Running simulation\n");
	
	anim = Glib::signal_timeout()
		.connect(sigc::mem_fun(*this,&Application::step), 100 );

//        anim.unblock();
	glArea->queue_draw();
}

void Application::create()
{
	anim.disconnect();

	int w = sp_width ->get_value_as_int(); 
	int h = sp_height->get_value_as_int();

	crystal.init( w , h );
	glArea->setZoom( w , h ) ;

	glArea->bufferResize( crystal.getAtomsBuffer() , crystal.getAtomsBufferSize() );
	glArea->bufferResize( crystal.getCrystalBuffer() , crystal.getCrystalBufferSize() );

//        pause();
	glArea->queue_draw();
}

void Application::pause()
{
	log_printf(DBG,"Pausing simulation\n");

//        anim.block();
	anim.disconnect();
}

