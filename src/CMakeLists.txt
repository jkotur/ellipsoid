cmake_policy(VERSION 2.6)

add_subdirectory( timer )
add_subdirectory( rng )

link_directories(
	${PROJECT_SOURCE_DIR}/lib/
	${GTKGL_LIBRARY_DIRS}
	${GTKMM_LIBRARY_DIRS}
	${GTHREAD_LIBRARY_DIRS}
	${OPENGL_LIBRARY_DIR}
	${GTKGLEXTMM_LIBRARY_DIRS} )

include_directories(
	${PROJECT_SOURCE_DIR}/include/
	${CUDA_INCLUDE_DIRS}
	${GTKGL_INCLUDE_DIRS}  
	${GTKMM_INCLUDE_DIRS}  
	${GTHREAD_INCLUDE_DIRS}  
	${OPENGL_INCLUDE_DIR}
	${GLEW_INCLUDE_PATH}
	${GTKGLEXTMM_INCLUDE_DIRS} )

CUDA_ADD_LIBRARY( cuda
	Crystal.cu 
	CrystalKernel.cu
	cuda_util
	)

add_library( gui
	GlDrawing
	GlUtils )

add_executable( raycasting
	main
	logger
	Application
	)

target_link_libraries( raycasting cuda gui timer rng 
	${CUDA_LIBRARIES}
	
	${GTKGL_LIBRARIES}
	${GTKMM_LIBRARIES}
	${GTHREAD_LIBRARIES}
	${OPENGL_LIBRARIES}
	${GLEW_LIBRARY}
	${GTKGLEXTMM_LIBRARIES} )

