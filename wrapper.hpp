#include "python_common.hpp"
#include "file.hpp"

struct py_wav_handle : PyObject, wav_file {
    py_wav_handle(const char* name) : wav_file(name) {}
};

extern PyTypeObject py_wav_handle_type;