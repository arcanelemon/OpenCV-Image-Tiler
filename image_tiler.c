#include <Python.h>
#include "image_tiler.h"
#include <unistd.h>

PyObject* split;
PyObject* tile;

void InitTiler()
{
    Py_Initialize();

    // Find Module
    char cwd[PATH_MAX]; 
    PySys_SetPath(getcwd(cwd, sizeof(cwd)));
 
    PyObject* pName = PyString_FromString("image_tiler.py"); // here
    PyObject* pModule = PyImport_Import(pName);
    PyObject* pDict = PyModule_GetDict(pModule);
    
    // grab funcions
    split = PyDict_GetItemString(pDict, "split");
    tile = PyDict_GetItemString(pDict, "tile");
    
    // clean up
    Py_DECREF(pName);
    Py_DECREF(pModule);
    Py_DECREF(pDict);
}

void CloseTiler()
{
    Py_DECREF(split);
    Py_DECREF(tile);
    Py_Finalize();
}

mat_cv** Split(mat_cv* image, int sliceSize)
{
    PyObject* pArgs = PyTuple_New(2);
    PyObject* pValue;

    // convert to ndarray
    PyObject* ndImage = ConvertMatToNDArray(image);

    // call object
    PyTuple_SetItem(pArgs, 0, ndImage);
    PyTuple_SetItem(pArgs, 1, PyLong_FromLong(sliceSize));
    pValue = PyObject_CallObject(split, pArgs);

    // convert to mat_cv
    int size = sizeof(pValue)/sizeof(pValue[0]);
    mat_cv** images[size];    
 
    for (int i = 0; i < size; i++)
    {
        mat_cv mat = ConvertNDArrayToMat(PyList_GetItem(pValue, i));
        images[i] = mat;
    }
   
    // clean up
    Py_DECREF(ndImage);
    Py_DECREF(pArgs);
    Py_DECREF(pValue);

    //return images;
}

mat_cv* Tile(mat_cv** images, int width, int height)
{
    PyObject* pArgs = PyTuple_New(3);
    PyObject* pValue;

    // convert to ndarray
    int size = sizeof(images)/sizeof(images[0]);
    PyObject** ndImages[size];    

    for (int i = 0; i < size; i++)
    {
        ndImages[i] = ConvertMatToNDArray(images[i]);
    }
    
    // call object
    PyTuple_SetItem(pArgs, 0, ndImages);
    PyTuple_SetItem(pArgs, 1, PyLong_FromLong(width));
    PyTuple_SetItem(pArgs, 2, PyLong_FromLong(height));
    pValue = PyObject_CallObject(tile, pArgs);

    // convert to mat_cv  
    mat_cv* image = ConvertNDArrayToMat(pValue); 
   
    // clean up
    Py_DECREF(ndImages);
    Py_DECREF(pArgs);
    Py_DECREF(pValue);

    //return image;
}

mat_cv* ConvertNDArrayToMat(PyObject* ndArray)
{
    PyArrayObject* contig = (PyArrayObject*)PyArray_FromAny(ndArray, PyArray_DescrFromType(NPY_UINT8),
    3, 3, NPY_DEFAULT, NULL);

    int rows = PyArray_DIM(contig, 0);
    int cols = PyArray_DIM(contig, 1);
    int channel = PyArray_DIM(contig, 2);
    int depth = cv2.CV_8U;
  
    int type = CV_MAKETYPE(depth, channel);
    mat_cv* mat = mat_cv(rows, cols, type);
    memcpy(mat.data, PyArray_DATA(contig), sizeof(uchar) * rows * cols * channel);
  
    //mat_cv mat = mat_cv(PyArray_DIM(contig, 0), PyArray_DIM(contig, 1), CV_8UC1);
    //memcpy(mat.data, PyArray_DATA(contig), sizeof(uchar) * mat.rows * mat.col )

    //int rows = shape[0];
    //int cols = shape[1];
    //int channel = shape[2];
    //int depth = CV_8U;

    //int type = CV_MAKETYPE(depth, channel);
    //mat_cv* image = mat_cv(rows, cols, type);
    //memcpy(image.data, ndArr.get_data(), sizeof(uchar) * rows * cols * channel);
   
    return &mat; 
}

PyObject* ConvertMatToNDArray(mat_cv* mat)
{
  PyTuple* shape = PyTuple_MakeTuple(mat.rows, mat.cols, mat.channels())
  PyTuple* stride = PyTuple_MakeTuple(mat.channels() * mat.cols * mat.sizeof(uchar), mat.channels() * sizeof(uchar), sizeof(uchar));
  dType dt = numpy.dtype.getBuildin<uchar>();
  ndarray ndImg = np.from_data(mat.data, dt, shape, stride, PyObject());
return ndImg
}

//COMMON+= `python2.7-config --cflags`
//LDFLAGS+= `/usr/bin/python2.7-config --ldflags`

