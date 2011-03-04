#ifndef __GLUTILS_H__

#define __GLUTILS_H__

//
// OpenGL frame buffer configuration utilities.
//

struct GLConfigUtil
{
  static void print_gl_attrib(const Glib::RefPtr<const Gdk::GL::Config>& glconfig,
                              const char* attrib_str,
                              int attrib,
                              bool is_boolean);

  static void examine_gl_attrib(const Glib::RefPtr<const Gdk::GL::Config>& glconfig);
};


#endif /* __GLUTILS_H__ */

