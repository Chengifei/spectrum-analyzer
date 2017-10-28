#include "python_common.hpp"
#include <cstddef>
#include "file.hpp"

template <typename T>
struct buffer {
    T* buf = nullptr;
    void resize(std::size_t sz) {
        buf = static_cast<T*>(realloc(buf, sz * sizeof(T)));
    }
    T* get() {
        return buf;
    }
    ~buffer() {
        free(buf);
    }
};

struct py_wav_handle : PyObject, wav_file {
    buffer<double> buf;
    Py_ssize_t len;
    Py_buffer pybuf;
    py_wav_handle(const char* name) : wav_file(name) {}
};

struct py_bmp_writer : PyObject, bmp_file {
    py_bmp_writer(int width, int height) : bmp_file(width, height) {};
};

extern PyTypeObject py_wav_handle_type;
extern PyTypeObject py_bmp_writer_type;