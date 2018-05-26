#include "wrapper.hpp"

static PyObject* wav_getattro(PyObject *self, PyObject *attro) {
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

static PyObject* tell(PyObject* self, PyObject*) {
    wav_file& s = *static_cast<py_wav_handle*>(self);
    return PyLong_FromLong(s.tell());
}

static int wav_file_init(PyObject* self, PyObject* args, PyObject*) {
    const char* name;
    if (!PyArg_ParseTuple(args, "s", &name))
        return -1;
    new(static_cast<wav_file*>(static_cast<py_wav_handle*>(self))) wav_file(name);
    return 0;
}

static PyMethodDef WavMethods[] = {
    { "jump_to", jump_to, METH_VARARGS, "" },
    { "inc", inc, METH_O, "" },
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
    wav_getattro,            /* tp_getattro */
    wav_setattro,            /* tp_setattro */
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