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
    Pa_Initialize();
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

PyObject* get_pitch(PyObject*, PyObject* args) {
    PyObject* py_wav;
    unsigned interval;
    if (!PyArg_ParseTuple(args, "OI", &py_wav, &interval))
        return nullptr;
    wav_file& wav = *static_cast<py_wav_handle*>(py_wav);
    const std::size_t len = wav.size_of(interval);
    Eigen::VectorXd ref(len, 1);
    if (wav.get(len, ref.data())) {
        PyErr_SetString(PyExc_IndexError, "File length exceeded");
        return nullptr;
    }
    // Fitting form x[i] = sum(a[j] * cos(omega[j] * n) + b[j] * sin(omega[j] * n), j)
    // where i and n are time, namely n = i / sample_rate
    //Eigen::MatrixXd cos_sin_consts(len, 88 * 2);
    //for (int i = 0; i != len; ++i) {
    //    // first 88 cos, second 88 sin
    //    for (int j = 0; j != 88; ++j) {
    //        double tmp = i * piano_omegas[j] / wav.sample_rate;
    //        cos_sin_consts(i, j) = std::cos(tmp);
    //        cos_sin_consts(i, 88 + j) = std::sin(tmp);
    //    }
    //}
    Eigen::MatrixXd tmp(2 * 88, 2 * 88); // = cos_sin_consts.transpose() * cos_sin_consts
    //for (int i = 0; i < 88; ++i) {
    //    for (int j = 0; j < 88; ++j) {
    //        double acc1 = 0, acc2 = 0, acc3 = 0, acc4 = 0;
    //        for (int k = 0; k != len; ++k) {
    //            double tmp = k * piano_omegas[i] / wav.sample_rate;
    //            double tmp2 = k * piano_omegas[j] / wav.sample_rate;
    //            acc1 += std::cos(tmp) * std::cos(tmp2);
    //            acc2 += std::cos(tmp) * std::sin(tmp2);
    //            acc3 += std::sin(tmp) * std::cos(tmp2);
    //            acc4 += std::sin(tmp) * std::sin(tmp2);
    //        }
    //        tmp(i, j) = acc1;
    //        tmp(i, 88 + j) = acc2;
    //        tmp(88 + i, j) = acc3;
    //        tmp(88 + i, 88 + j) = acc4;
    //    }
    //}
    // Lagrange's formulae applied and optimized for symmetricity
    for (int i = 0; i != 88; ++i) {
        double htmp3 = piano_omegas[i] / wav.sample_rate;
        double tmp3 = 2 * htmp3;
        tmp(i, i) = len - 0.5 + std::sin(len * tmp3 + htmp3) / (2 * std::sin(htmp3));
        tmp(i, 88 + i) = 0.5 * (std::cos(htmp3) - std::cos(len * tmp3 + htmp3)) / std::sin(htmp3);
        tmp(88 + i, 88 + i) = len + 0.5 - std::sin(len * tmp3 + htmp3) / (2 * std::sin(htmp3));
        for (int j = i + 1; j != 88; ++j) {
            double tmp1 = piano_omegas[i] / wav.sample_rate;
            double tmp2 = piano_omegas[j] / wav.sample_rate;
            double tmp3 = tmp1 + tmp2, tmp4 = tmp1 - tmp2;
            double htmp3 = tmp3 * 0.5, htmp4 = tmp4 * 0.5;
            tmp(i, j) = std::sin(len * tmp3 + htmp3) / (2 * std::sin(htmp3)) + std::sin(len * tmp4 + htmp4) / (2 * std::sin(htmp4)) - 1;
            tmp(i, 88 + j) = 0.5 * (std::cos(htmp3) - std::cos(len * tmp3 + htmp3)) / std::sin(htmp3) - 0.5 * (std::cos(htmp4) - std::cos(len * tmp4 + htmp4)) / std::sin(htmp4);
            /* tmp(88 + i, j) */ tmp(j, 88 + i) = 0.5 * (std::cos(htmp3) - std::cos(len * tmp3 + htmp3)) / std::sin(htmp3) + 0.5 * (std::cos(htmp4) - std::cos(len * tmp4 + htmp4)) / std::sin(htmp4);
            tmp(88 + i, 88 + j) = std::sin(len * tmp4 + htmp4) / (2 * std::sin(htmp4)) - std::sin(len * tmp3 + htmp3) / (2 * std::sin(htmp3));
        }
    }
    tmp /= 2;
    Eigen::Matrix<double, 88 * 2, 1> rhs; // = cos_sin_consts.tranpose() * ref
#pragma omp parallel for
    for (int i = 0; i < 88; ++i) {
        double acc0 = 0;
        double acc1 = 0, acc2 = 0;
        for (int j = 0; j != len; ++j) {
            acc1 += std::cos(acc0) * ref[j];
            acc2 += std::sin(acc0) * ref[j];
            acc0 += piano_omegas[i] / wav.sample_rate;
        }
        rhs[i] = acc1;
        rhs[88 + i] = acc2;
    }
    Eigen::Matrix<double, 88 * 2, 1> sol = tmp.selfadjointView<Eigen::Upper>().llt().solve(rhs);
    Eigen::Map<Eigen::Array<double, 88, 1>> a(sol.data());
    Eigen::Map<Eigen::Array<double, 88, 1>> b(sol.data() + 88);
    npy_intp dim[] = { 88 };
    PyObject* ret = PyArray_SimpleNew(1, dim, NPY_DOUBLE);
    if (!ret)
        return nullptr;
    Eigen::Map<Eigen::Array<double, 88, 1>> power(static_cast<double*>(PyArray_GETPTR1(ret, 0)));
    power = (a.cwiseAbs2() + b.cwiseAbs2());
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
