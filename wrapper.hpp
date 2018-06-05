#include "python_common.hpp"
#include "file.hpp"
#include <portaudio.h>

struct pa_stream_handle : PyObject {
    PaStream* stream;
    wav_file* file;
    std::size_t init;
    std::size_t len;
    ~pa_stream_handle();
};

struct py_wav_handle : PyObject, wav_file {
    pa_stream_handle* player;
};

extern PyTypeObject py_wav_handle_type;
extern PyTypeObject pa_stream_handle_type;
