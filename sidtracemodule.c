#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "arrayobject.h"

#include "math.h"
#define SIZE 20000

static PyObject *
sidtrace_trace(PyObject *self, PyObject *args)
{
    // INPUT
    double st,dx,dy;
    int nx,ny;
    PyObject  *x, *y, *kx, *ky,*ax,*ay;

    if (!PyArg_ParseTuple(args, "OOOOdddiiOO", &x,&y,&kx,&ky,&st,&dx,&dy,&nx,&ny,&ax,&ay))
        return NULL;

	//TRACE

	// iternal and service vars
	int i = 0;
	int x_i,y_i;
    double xx, yy ,kxx,kyy, Mx,My;
    void *data;
	PyObject *item;

    data = PyArray_GETPTR1(kx,0);
    item = PyArray_GETITEM(kx,data);
    kxx = PyFloat_AsDouble(item);

    data = PyArray_GETPTR1(ky,0);
    item = PyArray_GETITEM(ky,data);
    kyy = PyFloat_AsDouble(item);

    data = PyArray_GETPTR1(x,0);
    item = PyArray_GETITEM(x,data);
    xx = PyFloat_AsDouble(item);

    data = PyArray_GETPTR1(y,0);
    item = PyArray_GETITEM(y,data);
    yy = PyFloat_AsDouble(item);


   while (i<SIZE-1 && xx >= 0 && xx < nx-1 && yy >= 0 && yy< ny-1) 
   	{
	x_i = (int)floor(xx);
	y_i = (int)floor(yy);

        data = PyArray_GETPTR2(ax,y_i,x_i);
        item = PyArray_GETITEM(ax,data);
        Mx = PyFloat_AsDouble(item);

        data = PyArray_GETPTR2(ay,y_i,x_i);
        item = PyArray_GETITEM(ay,data);
        My = PyFloat_AsDouble(item);

        xx = xx + (st * st * Mx / 2 + kxx * st ) / dx;
        yy = yy + (st * st * My / 2 + kyy * st ) / dy;

        kxx = kxx + st * Mx;
        kyy = kyy + st * My;


        // store in Py vars
        i++;
        data = PyArray_GETPTR1(kx,i);
        PyArray_SETITEM(kx,data,PyFloat_FromDouble(kxx));

        data = PyArray_GETPTR1(ky,i);
        PyArray_SETITEM(ky,data,PyFloat_FromDouble(kyy));

        data = PyArray_GETPTR1(x,i);
        PyArray_SETITEM(x,data,PyFloat_FromDouble(xx));

        data = PyArray_GETPTR1(y,i);
        PyArray_SETITEM(y,data,PyFloat_FromDouble(yy));
	}

    return Py_BuildValue("i",i);
}
static PyObject *
sidtrace_check(PyObject *self, PyObject *args)
{
    // INPUT
    PyObject *alpha;
    int i,j;
    double k;

    if (!PyArg_ParseTuple(args, "Oiid", &alpha,&i,&j,&k))
      return NULL;

    //Extract values

    void *data;
    PyObject *item;
    double A;

    data = PyArray_GETPTR2(alpha,i,j);
    item = PyArray_GETITEM(alpha,data);
    A = k * PyFloat_AsDouble(item);
    PyArray_SETITEM(alpha,data,PyFloat_FromDouble(A));

    return Py_BuildValue("iid",i,j,k);
}
static PyMethodDef SidtraceMethods[] = {
    {"trace",  sidtrace_trace, METH_VARARGS, "Trace the ray"},
    {"check",  sidtrace_check, METH_VARARGS, "check reading"},
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