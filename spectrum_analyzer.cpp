#include "wrapper.hpp"
#include "fft.hpp"
#include "piano.hpp"
#include "python_common.hpp"
#include <algorithm>

static PyModuleDef Module = {
    PyModuleDef_HEAD_INIT,
    "spectrum_analyzer",
    "",
    -1,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr
};

PyObject* fft(PyObject*, PyObject* args) {
    PyObject* ar;
    unsigned num_of_freqs;
    if (!PyArg_ParseTuple(args, "OI", &ar, &num_of_freqs))
        return nullptr;
    PyOnly(PyMemoryView_Check(ar), true);
    Py_buffer* buf = PyMemoryView_GET_BUFFER(ar);
    PyOnly(strcmp(buf->format, "d"), 0);
    double* ptr = static_cast<double*>(buf->buf);
    std::size_t len = buf->len / buf->itemsize;
    std::unique_ptr<double[]> wsave = std::make_unique<double[]>(get_ret_size(len));
    npy_rffti(len, wsave.get());
    npy_rfftf(len, ptr, wsave.get());
    ptr[-1] = ptr[0];
    ptr[0] = 0;
    buf->buf = --ptr;
    for (unsigned i = 0; i < num_of_freqs; ++i)
        ptr[i] = std::log(std::pow(ptr[i * 2], 2) + std::pow(ptr[i * 2 + 1], 2));
    buf->len = num_of_freqs * sizeof(double);
    buf->shape = nullptr;
    PyObject* ret = PyMemoryView_FromBuffer(buf);
    PyObject_CallMethod(ar, "release", nullptr);
    return ret;
}

void raw_fft(double ar[], std::size_t len, double wsave[]) {
    npy_rffti(len, wsave);
    npy_rfftf(len, ar, wsave);
    ar[0] = ar[1];
    ar[1] = 0;
}

template <std::size_t n, typename T>
std::size_t get_n_top(T in[], std::size_t in_sz, std::size_t out[]) {
    std::size_t out_count = 0;
    std::size_t* min_ptr = nullptr;
    for (std::size_t i = 0; i != in_sz; ++i) {
        if (out_count != n)
            out[out_count++] = i;
        else if (in[i] > *min_ptr)
            *min_ptr = i;
        min_ptr = std::min_element(out, out + n,
                                   [in](std::size_t a, std::size_t b) {
            return in[a] < in[b];
        });
    }
    return out_count;
}

PyObject* try_pitch(PyObject*, PyObject* args) {
    PyObject* py_wav;
    unsigned interval;
    if (!PyArg_ParseTuple(args, "OI", &py_wav, &interval))
        return nullptr;
    wav_file& wav = *static_cast<wav_file*>(static_cast<py_wav_handle*>(py_wav));
    std::size_t len = wav.size_of(interval);
    std::unique_ptr<double[]> buf = std::make_unique<double[]>(len + 1);
    std::unique_ptr<double[]> wsave = std::make_unique<double[]>(get_ret_size(len));
    wav.get(interval, buf.get() + 1);
    raw_fft(buf.get() + 1, len, wsave.get());
    double freq_width = 1000 / double(interval);
    len = (piano_freqs[87] + freq_width) / freq_width;
    buf[0] = buf[0] ? std::log(buf[0]) * 2 : 0; // use this as DC reference
    double power[88];
    for (std::size_t i = 0; i != 88; ++i) {
        std::size_t offset = std::round(piano_freqs[i] / freq_width);
        power[i] = std::log(std::pow(buf[offset * 2], 2) + std::pow(buf[offset * 2 + 1], 2));
    }
    double factor = 255 / (*std::max_element(power, power + 88) - buf[0]);
    PyObject* ret = PyList_New(88);
    for (std::size_t i = 0; i != 88; ++i) {
        PyList_SetItem(ret, i, PyLong_FromLong((power[i] > buf[0]) ? factor * (power[i] - buf[0]) : 0));
    }
    return ret;
}

static PyMethodDef Methods[] = {
    { "fft", fft, METH_VARARGS, "" },
    { "get_pitch", try_pitch, METH_VARARGS, "" },
    { nullptr }
};

PyMODINIT_FUNC
PyInit_spectrum_analyzer() {
    PyObject* m = PyExc(PyModule_Create(&Module), nullptr);
    PyModule_AddFunctions(m, Methods);
    PyOnly(PyType_Ready(&py_wav_handle_type), 0);
    Py_INCREF(&py_wav_handle_type);
    PyModule_AddObject(m, "wav_file", reinterpret_cast<PyObject*>(&py_wav_handle_type));
    PyOnly(PyType_Ready(&py_bmp_writer_type), 0);
    Py_INCREF(&py_bmp_writer_type);
    PyModule_AddObject(m, "bmp_file", reinterpret_cast<PyObject*>(&py_bmp_writer_type));
    return m;
}