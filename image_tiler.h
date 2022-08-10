#ifndef IMAGE_TILER_H
#define IMAGE_TILER_H

#include "Python.h"
#include "network.h"

void InitTiler();
void CloseTiler();
mat_cv** Split(mat_cv* image, int sliceSize);
mat_cv* Tile(mat_cv** images, int width, int height);
mat_cv* ConvertNDArrayToMat(PyObject* ndArray);
PyObject* ConvertMatToNDArray(mat_cv* mat);

#endif
