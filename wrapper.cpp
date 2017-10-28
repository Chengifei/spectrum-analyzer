#include "wrapper.hpp"
#include <algorithm>

PyObject* cNVar_getattro(PyObject* self, PyObject* attro) {
    if (auto i = PyObject_GenericGetAttr(self, attro))
        return i;
    PyErr_Clear();
    const char* attr_name = PyExc(PyUnicode_AsUTF8(attro), nullptr);
    wav_file& s = *static_cast<wav_file*>(static_cast<py_wav_handle*>(self));
    if (!strcmp(attr_name, "channels"))
        return PyLong_FromLong(s.channels);
    if (!strcmp(attr_name, "sample_rate"))
        return PyLong_FromLong(s.sample_rate);
    PyErr_Format(PyExc_AttributeError,
                 "wav_file object has no attribute '%.400s'", attr_name);
    return nullptr;
}

int cNVar_setattro(PyObject*, PyObject*, PyObject*) {
    PyErr_SetString(PyExc_AttributeError,
                 "wav_handle object is readonly");
    return -1;
}

PyObject* jump_to(PyObject* self, PyObject* args) {
    unsigned sec, ms;
    if (!PyArg_ParseTuple(args, "II", &sec, &ms))
        return nullptr;
    wav_file& s = *static_cast<wav_file*>(static_cast<py_wav_handle*>(self));
    s.jump_to(sec, ms);
    Py_RETURN_NONE;
}

PyObject* inc(PyObject* self, PyObject* args) {
    unsigned ms;
    if (!PyArg_ParseTuple(args, "I", &ms))
        return nullptr;
    wav_file& s = *static_cast<wav_file*>(static_cast<py_wav_handle*>(self));
    s.inc(ms);
    Py_RETURN_NONE;
}

PyObject* tell(PyObject* self, PyObject*) {
    wav_file& s = *static_cast<wav_file*>(static_cast<py_wav_handle*>(self));
    return PyLong_FromLong(s.tell());
}

PyObject* get(PyObject* self, PyObject* args) {
    unsigned ms;
    if (!PyArg_ParseTuple(args, "I", &ms))
        return nullptr;
    Py_buffer& pybuf = static_cast<py_wav_handle*>(self)->pybuf;
    Py_ssize_t& len = static_cast<py_wav_handle*>(self)->len;
    buffer<double>& buf = static_cast<py_wav_handle*>(self)->buf;
    wav_file& s = *static_cast<wav_file*>(static_cast<py_wav_handle*>(self));
    if (std::size_t req_len = s.size_of(ms) + 1; len < req_len) {
        buf.resize(req_len);
        static_cast<py_wav_handle*>(self)->len = req_len;
        pybuf.len = (req_len - 1) * sizeof(double);
        pybuf.buf = buf.get() + 1;
    }
    s.get(ms, buf.get() + 1);
    return PyMemoryView_FromBuffer(&pybuf);
}

int wav_file_init(PyObject* self, PyObject* args, PyObject*) {
    const char* name;
    if (!PyArg_ParseTuple(args, "s", &name))
        return -1;
    new(static_cast<wav_file*>(static_cast<py_wav_handle*>(self))) wav_file(name);
    Py_buffer& pybuf = static_cast<py_wav_handle*>(self)->pybuf;
    Py_INCREF(self);
    pybuf.obj = self;
    pybuf.len = 0;
    pybuf.itemsize = sizeof(double);
    pybuf.readonly = true;
    pybuf.ndim = 1;
    pybuf.format = "d";
    new(&static_cast<py_wav_handle*>(self)->buf) buffer<double>();
    return 0;
}

static PyMethodDef WavMethods[] = {
    { "jump_to", jump_to, METH_VARARGS, "" },
    { "inc", inc, METH_VARARGS, "" },
    { "get", get, METH_VARARGS, "" },
    { "tell", tell, METH_NOARGS, "" },
    { nullptr }
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
    cNVar_getattro,            /* tp_getattro */
    cNVar_setattro,            /* tp_setattro */
    0,                         /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,        /* tp_flags */
    0,                         /* tp_doc */
    0,                         /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    0,                         /* tp_iter */
    0,                         /* tp_iternext */
    WavMethods,                   /* tp_methods */
    0,                         /* tp_members */
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

int bmp_file_init(PyObject* self, PyObject* args, PyObject*) {
    int w, h;
    if (!PyArg_ParseTuple(args, "ii", &w, &h))
        return -1;
    new(static_cast<bmp_file*>(static_cast<py_bmp_writer*>(self))) bmp_file(w, h);
    return 0;
}

PyObject* save(PyObject* self, PyObject* args) {
    const char* str;
    if (!PyArg_ParseTuple(args, "s", &str))
        return nullptr;
    bmp_file& s = *static_cast<bmp_file*>(static_cast<py_bmp_writer*>(self));
    s.save(str);
    Py_RETURN_NONE;
}

PyObject* use_array(PyObject* self, PyObject* args) {
    PyObject* view;
    if (!PyArg_ParseTuple(args, "O", &view))
        return nullptr;
    bmp_file& s = *static_cast<bmp_file*>(static_cast<py_bmp_writer*>(self));
    PyOnly(PyMemoryView_Check(view), true);
    Py_buffer* buf = PyMemoryView_GET_BUFFER(view);
    PyOnly(strcmp(buf->format, "d"), 0);
    double* ptr = static_cast<double*>(buf->buf);
    std::size_t len = buf->len / buf->itemsize;
    double factor = 256 / *std::max_element(ptr, ptr + len);
    std::unique_ptr<std::uint8_t[]> char_ar = std::make_unique<std::uint8_t[]>(len);
    for (std::size_t n = 0; n != len; ++n)
        char_ar[n] = std::uint8_t(factor * ptr[n]);
    s.use_array(char_ar.get(), len);
    Py_RETURN_NONE;
}

static PyMethodDef BmpMethods[] = {
    { "save", save, METH_VARARGS, "" },
    { "array", use_array, METH_VARARGS, "" },
    { nullptr }
};

PyTypeObject py_bmp_writer_type {
    PyVarObject_HEAD_INIT(NULL, 0)
    "bmp_file",                /* tp_name */
    sizeof(py_bmp_writer),     /* tp_basicsize */
    0,                         /* tp_itemsize */
    call_destructor<py_bmp_writer>,/* tp_dealloc */
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
    cNVar_setattro,            /* tp_setattro */
    0,                         /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,        /* tp_flags */
    0,                         /* tp_doc */
    0,                         /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    0,                         /* tp_iter */
    0,                         /* tp_iternext */
    BmpMethods,                   /* tp_methods */
    0,                         /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    bmp_file_init,             /* tp_init */
    0,                         /* tp_alloc */
    PyType_GenericNew          /* tp_new */
};
