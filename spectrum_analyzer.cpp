#include "wrapper.hpp"
#include "fft.hpp"
#include "python_common.hpp"

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

static PyMethodDef Methods[] = {
    { "fft", fft, METH_VARARGS, "" },
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