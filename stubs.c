
#include <dlfcn.h>
#include <stddef.h>
void* __wrap_dlopen(const char* filename, int flags) {
    // Intercept attempts to load the libraries we want to exclude
    if (filename && (
        strstr(filename, "libblas.so") ||
        strstr(filename, "libGLX.so") ||
        strstr(filename, "liblapack.so"))) {
        return NULL;  // Return NULL to indicate the library couldn't be loaded
    }
    // Call the real dlopen for all other libraries
    return dlopen(filename, flags);
}
