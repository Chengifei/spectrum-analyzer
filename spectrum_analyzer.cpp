#include "wrapper.hpp"
#include "piano.hpp"
#include "python_common.hpp"
#include <Eigen/Dense>
#include <numpy/ndarrayobject.h>
#include <portaudio.h>

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

int PaInit(PyObject*, PyObject*, PyObject*) {
    // We don't acquire GIL here becuase portaudio won't access Python objects.
    if (auto err = Pa_Initialize(); err != paNoError) {
        PyErr_SetString(PyExc_SystemError, Pa_GetErrorText(err));
        return -1;
    }
    return 0;
}

void PaEnd(PyObject*) {
    Pa_Terminate();
}

PyTypeObject PaInitializerType {
    PyVarObject_HEAD_INIT(NULL, 0)
    "_PaInitializer",          /* tp_name */
    sizeof(PyObject),          /* tp_basicsize */
    0,                         /* tp_itemsize */
    PaEnd,                     /* tp_dealloc */
    0,                         /* tp_print */
    0,                         /* tp_getattr */
    0,                         /* tp_setattr */
    0,                         /* tp_reserved */
    0,                         /* tp_repr */
    0,                         /* tp_as_number */
    0,                         /* tp_as_sequence */
    0,                         /* tp_as_mapping */
    0,                         /* tp_hash  */
    0,                         /* tp_call */
    0,                         /* tp_str */
    0,                         /* tp_getattro */
    0,                         /* tp_setattro */
    0,                         /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,        /* tp_flags */
    0,                         /* tp_doc */
    0,                         /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    0,                         /* tp_iter */
    0,                         /* tp_iternext */
    0,                         /* tp_methods */
    0,                         /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    PaInit,                    /* tp_init */
    0,                         /* tp_alloc */
    PyType_GenericNew          /* tp_new */
};

constexpr std::size_t n_peel[] = {
    73, 61, 54, 49, 46, 42, 40, 37, 35, 34, 32, 30, 29, 28, 27, 25, 24, 23, 23, 22, 21, 20, 19, 18, 18, 17, 16, 16, 15, 15, 14, 13, 13, 12, 12, 11, 11, 11, 10, 10, 9, 9, 8, 8, 8, 7, 7, 6, 6, 6, 5, 5, 5, 4, 4, 4, 3, 3, 3, 3, 2, 2, 2, 1, 1, 1, 1
};

PyObject* get_pitch(PyObject*, PyObject* args) {
    PyObject* py_wav;
    unsigned interval;
    if (!PyArg_ParseTuple(args, "OI", &py_wav, &interval))
        return nullptr;
    wav_file& wav = *static_cast<py_wav_handle*>(py_wav);
    const std::size_t len = wav.size_of(interval);
    const std::size_t peel_idx = (interval + 4) / 5;
    const std::size_t freq_peel = (peel_idx < sizeof n_peel / sizeof(std::size_t)) ? n_peel[peel_idx] : 0;
    Eigen::VectorXd ref(len, 1);
    if (wav.get(len, ref.data())) {
        PyErr_SetString(PyExc_IndexError, "File length exceeded");
        return nullptr;
    }
    Eigen::MatrixXd tmp(len, (88 - freq_peel) * 2 + 1);
    double rcp = 1. / wav.sample_rate; // reciprocal
    for (std::size_t i = 0; i != 88 - freq_peel; ++i) {
        for (std::size_t j = 0; j != len; ++j) {
            const double time = j * rcp;
            tmp(j, i) = std::sin(piano_omegas[freq_peel + i] * time);
            tmp(j, 88 - freq_peel + i) = std::cos(piano_omegas[freq_peel + i] * time);
        }
    }
    tmp.col((88 - freq_peel) * 2).setConstant(1);
    Eigen::VectorXd sol = tmp.fullPivHouseholderQr().solve(ref);
    Eigen::Map<Eigen::Array<double, Eigen::Dynamic, 2, Eigen::ColMajor>> a(sol.data(), (88 - freq_peel), 2);
    npy_intp dim[] = { 88 };
    PyObject* ret = PyArray_SimpleNew(1, dim, NPY_DOUBLE);
    if (!ret)
        return nullptr;
    Eigen::Map<Eigen::Array<double, 88, 1>> power(static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(ret))));
    power.topRows(freq_peel).setZero();
    power.bottomRows(88 - freq_peel) = (a.col(0).cwiseAbs2() + a.col(1).cwiseAbs2());
    return ret;
}

static PyMethodDef Methods[] = {
    { "get_pitch", get_pitch, METH_VARARGS, "" },
    { nullptr }
};

PyMODINIT_FUNC
PyInit_spectrum_analyzer() {
    PyObject* m = PyExc(PyModule_Create(&Module), nullptr);
    import_array();
    PyModule_AddFunctions(m, Methods);
    PyOnly(PyType_Ready(&py_wav_handle_type), 0);
    Py_INCREF(&py_wav_handle_type);
    PyModule_AddObject(m, "wav_file", reinterpret_cast<PyObject*>(&py_wav_handle_type));
    PyOnly(PyType_Ready(&PaInitializerType), 0);
    Py_INCREF(&PaInitializerType);
    PyOnly(PyType_Ready(&pa_stream_handle_type), 0);
    Py_INCREF(&pa_stream_handle_type);
    auto empty_arg = PyTuple_New(0);
    auto init = PyObject_CallObject(reinterpret_cast<PyObject*>(&PaInitializerType), empty_arg);
    Py_DECREF(empty_arg);
    PyModule_AddObject(m, "__Do_NOT_TouchThis", init);
    return m;
}
