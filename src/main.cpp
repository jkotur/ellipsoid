#include <string>
#include <iostream>
#include <cstdlib>
#include <ctime>

#include <GL/glew.h>
#include <GL/gl.h>

#include <gtkmm.h>
#include <gtkglmm.h>

#include <cuda.h>
#include <cudaGL.h>

#include "logger.h"
#include "cuda_util.h"

#include "Application.h"

const std::string gui_file = "torrusador.ui";

int main (int argc, char* argv[])
{
	std::srand( std::time(NULL) );

	FILE*f_log = fopen("path_finder.log","w");       
	log_add( LOG_STREAM(f_log)  , LOG_PRINTER(vfprintf) );
	log_add( LOG_STREAM(stderr) , LOG_PRINTER(vfprintf) );

//        log_set_lev(INFO);

	Glib::thread_init();
	Gtk::Main gtk(argc,argv);
	Gtk::GL::init(argc, argv);

	CUT_DEVICE_QUERY();

	Application app(gui_file);

	app.show();

	Gtk::Main::run();

	fclose(f_log);

	return 0;
}

