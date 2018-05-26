#include "python_common.hpp"
#include "file.hpp"

struct py_wav_handle : PyObject, wav_file {};

extern PyTypeObject py_wav_handle_type;