#include "wrapper.hpp"
#include "piano.hpp"
#include "python_common.hpp"
#include <Eigen/Dense>

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

PyObject* get_pitch(PyObject*, PyObject* args) {
    PyObject* py_wav;
    unsigned interval;
    if (!PyArg_ParseTuple(args, "OI", &py_wav, &interval))
        return nullptr;
    wav_file& wav = *static_cast<py_wav_handle*>(py_wav);
    const std::size_t len = wav.size_of(interval);
    Eigen::VectorXf ref(len, 1);
    wav.get(interval, ref.data());
	float dc_offset = ref.sum() / len;
	printf("%f\n", dc_offset);
	ref.array() -= dc_offset;
    // Fitting form x[i] = sum(a[j] * cos(omega[j] * n) + b[j] * sin(omega[j] * n), j)
    // where i and n are time, namely n = i / sample_rate
    Eigen::MatrixXf cos_sin_consts(len, 88 * 2);
    for (int i = 0; i != len; ++i) {
        float time = float(i) / wav.sample_rate;
        // first 88 cos, second 88 sin
        for (int j = 0; j != 88; ++j) {
            cos_sin_consts(i, j) = std::cos(time * piano_omegas[j]);
            cos_sin_consts(i, 88 + j) = std::sin(time * piano_omegas[j]);
        }
    }
    Eigen::Matrix<float, 88 * 2, 1> sol = cos_sin_consts.colPivHouseholderQr().solve(ref);
    Eigen::Map<Eigen::Array<float, 88, 1>> a(sol.data());
    Eigen::Map<Eigen::Array<float, 88, 1>> b(sol.data() + 88);
    Eigen::Array<float, 88, 1> power = (a.cwiseAbs2() + b.cwiseAbs2()).log();
    PyObject* ret = PyList_New(88);
    for (std::size_t i = 0; i != 88; ++i) {
        PyList_SetItem(ret, i, PyFloat_FromDouble(power[i]));
    }
    return ret;
}

static PyMethodDef Methods[] = {
    { "get_pitch", get_pitch, METH_VARARGS, "" },
    { nullptr }
};

PyMODINIT_FUNC
PyInit_spectrum_analyzer() {
    PyObject* m = PyExc(PyModule_Create(&Module), nullptr);
    PyModule_AddFunctions(m, Methods);
    PyOnly(PyType_Ready(&py_wav_handle_type), 0);
    Py_INCREF(&py_wav_handle_type);
    PyModule_AddObject(m, "wav_file", reinterpret_cast<PyObject*>(&py_wav_handle_type));
    return m;
}
