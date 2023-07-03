#include "wrapper.hpp"
#include <numpy/ndarrayobject.h>
#include <structmember.h>
extern "C" {
#include <libswresample/swresample.h>
}

static int portaudio_callback(const void*, void* out, unsigned long frame_count, const PaStreamCallbackTimeInfo*, PaStreamCallbackFlags, void* data) {
    pa_stream_handle& s = *static_cast<pa_stream_handle*>(data);
    if (s.len <= frame_count) {
        if (s.file->get(s.len, static_cast<uint16_t*>(out))) {
            s.file->jump_to(s.init);
            return paAbort;
        }
        s.file->jump_to(s.init);
        return paComplete;
    }
    else {
        if (s.file->get(frame_count, static_cast<uint16_t*>(out))) {
            s.file->jump_to(s.init);
            return paAbort;
        }
        s.file->inc_raw(frame_count);
        s.len -= frame_count;
        return paContinue;
    }
}

static int wav_setattro(PyObject*, PyObject*, PyObject*) {
    PyErr_SetString(PyExc_AttributeError,
        "wav_handle object is readonly");
    return -1;
}

static PyObject* jump_to(PyObject* self, PyObject* args) {
    unsigned ms;
    if (!PyArg_ParseTuple(args, "I", &ms))
        return nullptr;
    wav_file& s = *static_cast<wav_file*>(static_cast<py_wav_handle*>(self));
    s.jump_to(ms);
    Py_RETURN_NONE;
}

static PyObject* inc(PyObject* self, PyObject* arg) {
    unsigned ms = PyLong_AS_LONG(arg);
    wav_file& s = *static_cast<py_wav_handle*>(self);
    s.inc(ms);
    Py_RETURN_NONE;
}

static PyObject* get(PyObject* self, PyObject* arg) {
    unsigned ms = PyLong_AS_LONG(arg);
    wav_file& s = *static_cast<py_wav_handle*>(self);
    npy_intp frames = s.size_of(ms);
    auto ret = PyArray_SimpleNew(1, &frames, NPY_FLOAT);
    s.get(frames, static_cast<float*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(ret))));
    return ret;
}

static PyObject* tell(PyObject* self, PyObject*) {
    wav_file& s = *static_cast<py_wav_handle*>(self);
    return PyLong_FromLong(s.tell());
}

static int wav_file_init(PyObject* self, PyObject* args, PyObject*) {
#ifndef _WIN32
    const char* name;
    if (!PyArg_ParseTuple(args, "s", &name))
        return -1;
#else
    PyObject* pname;
    if (!PyArg_ParseTuple(args, "O", &pname))
        return -1;
    wchar_t name[128];
    PyUnicode_AsWideChar(pname, name, 128);
#endif
    import_array();
    new(static_cast<wav_file*>(static_cast<py_wav_handle*>(self))) wav_file(name);
    py_wav_handle& wav = *static_cast<py_wav_handle*>(self);
    wav.player = static_cast<pa_stream_handle*>(pa_stream_handle_type.tp_new(&pa_stream_handle_type, nullptr, nullptr));
    auto err = Pa_OpenDefaultStream(&wav.player->stream, 0, wav.channels, paInt16, wav.sample_rate, paFramesPerBufferUnspecified, portaudio_callback, wav.player);
    wav.player->file = static_cast<py_wav_handle*>(self);
    if (err != paNoError)
        return -1;
    return 0;
}

pa_stream_handle::~pa_stream_handle() {
    Pa_CloseStream(stream);
}

static PyObject* play(PyObject* self, PyObject* len) {
    unsigned ms = static_cast<unsigned>(PyLong_AsLong(len));
    py_wav_handle& wav = *static_cast<py_wav_handle*>(self);
    auto stream = static_cast<py_wav_handle*>(self)->player->stream;
    Pa_AbortStream(stream);
    wav.player->init = wav.tell();
    wav.player->len = wav.size_of(ms);
    auto err = Pa_StartStream(stream);
    if (err != paNoError)
        goto error;
    Py_RETURN_NONE;
error:
    PyErr_SetString(PyExc_SystemError, Pa_GetErrorText(err));
    return nullptr;
}

static PyObject* resample(PyObject* self, PyObject* args) {
    py_wav_handle& wav = *static_cast<py_wav_handle*>(self);
    unsigned len, nsample_rate;
    if (!PyArg_ParseTuple(args, "II", &len, &nsample_rate))
        return nullptr;
    if (wav.channels != 1) {
        PyErr_SetString(PyExc_ValueError, "Channel is not 1");
        return nullptr;
    }
    AVSampleFormat infmt = AV_SAMPLE_FMT_S16;
    SwrContext* ctx = swr_alloc_set_opts(nullptr, AV_CH_LAYOUT_MONO, AV_SAMPLE_FMT_S16, nsample_rate, AV_CH_LAYOUT_MONO, infmt, wav.sample_rate, 0, nullptr);
    swr_init(ctx);
    std::size_t size = wav.size_of(len);
    npy_intp outsz = swr_get_out_samples(ctx, size); // max
    PyObject* ret = PyArray_SimpleNew(1, &outsz, NPY_INT16);
    uint8_t* buf = static_cast<uint8_t*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(ret)));
    const uint8_t* in = static_cast<const uint8_t*>(wav.get());
    npy_intp actual_out_sz = swr_convert(ctx, &buf, outsz, &in, size);
    PyArray_Dims dims { &actual_out_sz, 1 };
    PyArray_Resize(reinterpret_cast<PyArrayObject*>(ret), &dims, true, NPY_ANYORDER);
    swr_free(&ctx);
    return ret;
}

static PyMethodDef WavMethods[] = {
    { "jump_to", jump_to, METH_VARARGS, "" },
    { "inc", inc, METH_O, "" },
    { "tell", tell, METH_NOARGS, "" },
    { "play", play, METH_O, "" },
    { "get", get, METH_O, "" },
    { "resample", resample, METH_VARARGS, "" },
    { nullptr }
};

static PyMemberDef WavMembers[] = {
    { "channels", T_SHORT, offsetof(py_wav_handle, channels), READONLY},
    { "sample_rate", T_UINT, offsetof(py_wav_handle, sample_rate), READONLY },
    { "bit_depth", T_SHORT, offsetof(py_wav_handle, bit_depth), READONLY },
    { "player", T_OBJECT_EX, offsetof(py_wav_handle, player), READONLY },
    {nullptr}
};

PyTypeObject py_wav_handle_type {
    PyVarObject_HEAD_INIT(NULL, 0)
    "wav_file",                /* tp_name */
    sizeof(py_wav_handle),     /* tp_basicsize */
    0,                         /* tp_itemsize */
    call_destructor<py_wav_handle>,/* tp_dealloc */
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
    PyObject_GenericGetAttr,   /* tp_getattro */
    wav_setattro,              /* tp_setattro */
    0,                         /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,        /* tp_flags */
    0,                         /* tp_doc */
    0,                         /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    0,                         /* tp_iter */
    0,                         /* tp_iternext */
    WavMethods,                /* tp_methods */
    WavMembers,                /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    wav_file_init,             /* tp_init */
    0,                         /* tp_alloc */
    PyType_GenericNew          /* tp_new */
};

PyTypeObject pa_stream_handle_type {
    PyVarObject_HEAD_INIT(NULL, 0)
    "stream_handle",                /* tp_name */
    sizeof(pa_stream_handle),     /* tp_basicsize */
    0,                         /* tp_itemsize */
    call_destructor<pa_stream_handle>,/* tp_dealloc */
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
    0,                         /* tp_init */
    0,                         /* tp_alloc */
    PyType_GenericNew          /* tp_new */
};
