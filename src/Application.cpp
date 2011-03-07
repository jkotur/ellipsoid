#include "Application.h"

#include <string>

#include <GL/glew.h>
#include <GL/gl.h>

#include <gtkmm.h>
#include <gtkglmm.h>

#include <vector_functions.h>

#include "logger.h"

#include "constants.h"

#include "GlDrawing.h"
#include "GlUtils.h"

Application::Application( const std::string& ui_file )
	: renderer(1,1,1,1)
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
	refBuilder->get_widget( "sp_dt" , sp_dt );

	refBuilder->get_widget( "rbut_xy" , rbut_xy );
	refBuilder->get_widget( "rbut_xz" , rbut_xz );
	refBuilder->get_widget( "rbut_yz" , rbut_yz );

	refBuilder->get_widget( "rbut_trans"    , rbut_trans    );
	refBuilder->get_widget( "rbut_scale"    , rbut_scale    );
	refBuilder->get_widget( "rbut_rotate"   , rbut_rotate   );
	refBuilder->get_widget( "rbut_isoscale" , rbut_isoscale );

	sp_a->set_value(1.0f);
	sp_b->set_value(1.0f);
	sp_c->set_value(1.0f);
	sp_m->set_value(1.0f);

	sp_dt->set_value(0.002f);

	sp_a->signal_value_changed()
		.connect( sigc::mem_fun(*this,&Application::on_a_changed) );
	sp_b->signal_value_changed()
		.connect( sigc::mem_fun(*this,&Application::on_b_changed) );
	sp_c->signal_value_changed()
		.connect( sigc::mem_fun(*this,&Application::on_c_changed) );
	sp_m->signal_value_changed()
		.connect( sigc::mem_fun(*this,&Application::on_m_changed) );
	sp_dt->signal_value_changed()
		.connect( sigc::mem_fun(*this,&Application::on_dt_changed) );

	but_quit->signal_clicked()
		.connect( sigc::mem_fun(*this,&Application::quit) );

	glArea = NULL;
	refBuilder->get_widget_derived("drawing_gl",glArea);
	glArea->setPixBuff( renderer.getPixBuff()   ); 
	glArea->setRenderer( &renderer );

	glArea->set_events( Gdk::BUTTON_PRESS_MASK | Gdk::BUTTON_RELEASE_MASK | Gdk::BUTTON1_MOTION_MASK);
	glArea->signal_motion_notify_event()
		.connect( sigc::mem_fun(*this,&Application::on_motion) );
	glArea->signal_button_press_event()
		.connect( sigc::mem_fun(*this,&Application::on_button_press) );
}

Application::~Application()
{
}

bool Application::on_button_press(GdkEventButton* event)
{
	if( event->button != 1 )
		return true;
	base_x = event->x;
	base_y =-event->y;
}

bool Application::on_motion( GdkEventMotion* event )
{
	float3 diff;
	float3 axis1;
	float3 axis2;

	     if( rbut_xy->get_active() ) {
		diff = make_float3( base_x - event->x , base_y + event->y , 0 );
		axis1 = make_float3( 0 , 1 , 0 );
		axis2 = make_float3( 1 , 0 , 0 );
	}
	else if( rbut_xz->get_active() ) {
		diff = make_float3( base_x - event->x , 0 , base_y + event->y );
		axis1 = make_float3( 0 , 0 , 1 );
		axis2 = make_float3( 1 , 0 , 0 );
	}
	else if( rbut_yz->get_active() ) {
		diff = make_float3( 0 , base_x - event->x , base_y + event->y );
		axis1 = make_float3( 0 , 1 , 0 );
		axis2 = make_float3( 0 , 0 , 1 );
	}

	     if( rbut_trans->get_active() ) 
		renderer.translate( diff.x * .01 , diff.y * .01 , diff.z * .01 );
	else if( rbut_scale->get_active() )
		renderer.scale( 1 + diff.x * .01 , 1 + diff.y * .01 , 1 + diff.z * .01 );
	else if( rbut_isoscale->get_active() ) {
		float s = 1.0f + .01f * ( diff.x + diff.y + diff.z );
		renderer.scale( s , s , s );
	}
	else if( rbut_rotate->get_active() ) {
		renderer.rotate(-(base_x-event->x) * .001f , axis1.x , axis1.y , axis1.z );
		renderer.rotate( (base_y+event->y) * .001f , axis2.x , axis2.y , axis2.z );
	}
	else return true;

	base_x = event->x;
	base_y =-event->y;

	renderer.reset();
	glArea->refresh();

	return true;
}

void Application::on_a_changed()
{
	renderer.set_a( sp_a->get_value() );
	glArea->refresh();
}

void Application::on_b_changed()
{
	renderer.set_b( sp_b->get_value() );
	glArea->refresh();
}

void Application::on_c_changed()
{
	renderer.set_c( sp_c->get_value() );
	glArea->refresh();
}

void Application::on_m_changed()
{
	renderer.set_m( sp_m->get_value() );
	glArea->refresh();
}

void Application::on_dt_changed()
{
	glArea->set_timeout( sp_dt->get_value() );
	glArea->refresh();
}

