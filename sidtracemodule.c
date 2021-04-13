#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "arrayobject.h"


static PyObject *
sidtrace_trace(PyObject *self, PyObject *args)
{
    // INPUT
    double st,dx,dy;
    int nx,ny;
    PyObject  *x, *y, *kx, *ky,*ax,*ay;

    if (!PyArg_ParseTuple(args, "OOOOdddiiOO", &x,&y,&kx,&ky,&st,&dx,&dy,&nx,&ny,&ax,&ay))
        return NULL;

    //represent as dynamic linear arrays
    double *xx, *yy ,*kxx, *kyy, *Mx,*My;

    kxx = (double*)PyArray_DATA(kx);
    kyy = (double*)PyArray_DATA(ky);

    xx = (double*)PyArray_DATA(x);
    yy = (double*)PyArray_DATA(y);

    Mx = (double*)PyArray_DATA(ax);
    My = (double*)PyArray_DATA(ay);

    // internal pointers and service vars
    int i = 0, ind,MaxI;
    MaxI= *PyArray_DIMS(x);

    //TRACE
    while (i<MaxI-1 && xx[i] >= 0 && xx[i] < nx-1 && yy[i] >= 0 && yy[i]< ny-1)
    {
	    ind = (int)yy[i]*(nx-1)+(int)xx[i];
	    // works well in any case the variables should be casted so truncating is not necessary

        xx[i+1] = xx[i] + (st * st * Mx[ind] / 2 + kxx[i] * st ) / dx;
        yy[i+1] = yy[i] + (st * st * My[ind] / 2 + kyy[i] * st ) / dy;

        kxx[i+1] = kxx[i] + st * Mx[ind];
        kyy[i+1] = kyy[i] + st * My[ind];

        i++;
    }

    return Py_BuildValue("i",i);
}

static PyObject *
sidtrace_check(PyObject *self, PyObject *args)
{
    int nx,ny,i,j;
    PyObject  *ax;

    if (!PyArg_ParseTuple(args, "iiiiO",&nx,&ny,&i,&j,&ax))
        return NULL;

    //represent as dynamic linear arrays
    double *a;
    a = (double*)PyArray_DATA(ax);

	int ind = i*nx+j;

    return PyFloat_FromDouble(a[ind]);
}

static PyMethodDef SidtraceMethods[] = {
    {"trace",  sidtrace_trace, METH_VARARGS, "Trace the ray"},
    {"check",  sidtrace_check, METH_VARARGS, "Check array access"},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

static struct PyModuleDef sidtracemodule = {
    PyModuleDef_HEAD_INIT,
    "sidtrace",   /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    SidtraceMethods
};

PyMODINIT_FUNC
PyInit_sidtrace(void)
{
    return PyModule_Create(&sidtracemodule);
}