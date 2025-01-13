#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

static PyObject* quantize(PyObject* self, PyObject* args) {
    PyObject* weights_obj;

    if (!PyArg_ParseTuple(args, "O", &weights_obj)) {
        return NULL;
    }

    PyArrayObject* weights_array = (PyArrayObject*)PyArray_FROM_OTF(weights_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    if (weights_array == NULL) {
        PyErr_SetString(PyExc_TypeError, "Input must be a NumPy array of type float32.");
        return NULL;
    }

    float* weights = (float*)PyArray_DATA(weights_array);
    int ndim = PyArray_NDIM(weights_array);
    npy_intp* shape = PyArray_SHAPE(weights_array);
    int size = PyArray_SIZE(weights_array);

    if (size == 0) {
        Py_DECREF(weights_array);
        PyErr_SetString(PyExc_ValueError, "Input array cannot be empty.");
        return NULL;
    }

    float min_val = weights[0], max_val = weights[0];
    for (int i = 1; i < size; i++) {
        if (weights[i] < min_val) min_val = weights[i];
        if (weights[i] > max_val) max_val = weights[i];
    }

    if (max_val == min_val) {
        Py_DECREF(weights_array);
        PyErr_SetString(PyExc_ValueError, "Input array must have a non-zero range.");
        return NULL;
    }

    float scale = 255.0 / (max_val - min_val);
    int zero_point = (int)(-min_val * scale) - 128;

    PyArrayObject* quantized_array = (PyArrayObject*)PyArray_ZEROS(ndim, shape, NPY_INT8, 0);
    if (quantized_array == NULL) {
        Py_DECREF(weights_array);
        PyErr_NoMemory();
        return NULL;
    }

    int8_t* quantized_data = (int8_t*)PyArray_DATA(quantized_array);
    for (int i = 0; i < size; i++) {
        quantized_data[i] = (int8_t)(round(weights[i] * scale) + zero_point);
    }

    PyObject* result = PyTuple_Pack(3, (PyObject*)quantized_array, PyFloat_FromDouble(scale), PyLong_FromLong(zero_point));

    Py_DECREF(weights_array);

    return result;
}

static PyMethodDef QuantizationMethods[] = {
    {"quantize", quantize, METH_VARARGS, "Quantize FP32 weights/activations to INT8."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef quantizationmodule = {
    PyModuleDef_HEAD_INIT,
    "quantization",
    NULL,
    -1,
    QuantizationMethods
};

PyMODINIT_FUNC PyInit_quantization(void) {
    import_array();
    return PyModule_Create(&quantizationmodule);
}
