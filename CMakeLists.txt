project( raycasting )

cmake_minimum_required(VERSION 2.8)

SET( CMAKE_MODULE_PATH
	${CMAKE_MODULE_PATH}
	${CMAKE_CURRENT_SOURCE_DIR}/cmake/ )

SET( EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin )

SET(CUDA_BUILD_EMULATION false)

find_package( CUDA   )
find_package( GLEW   )
find_package( OpenGL )

find_package(PkgConfig)
PKG_CHECK_MODULES(GTKGL REQUIRED gtkglextmm-1.2)
pkg_check_modules(GTKMM REQUIRED gtkmm-2.4)
pkg_check_modules(GTHREAD REQUIRED gthread-2.0)
pkg_check_modules(GLIBMM REQUIRED glibmm-2.4)

if( UNIX ) 
	set( CMAKE_C_FLAGS "-std=gnu99 -g -Wall -pedantic")
endif( UNIX )

set( CUDA_NVCC_FLAGS "-keep;-arch=sm_11;--ptxas-options=-v" )

add_subdirectory( src )

