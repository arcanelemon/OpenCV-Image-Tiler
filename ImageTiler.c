#define BOOST_PYTHON_STATIC_LIB
#define BOOST_LIB_NAME "boost_numpy35"
//#include <boost/config/auto_link.hpp>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>

namespace py = boost::python;
namespace np = boost::python::numpy;

public void Init() {
    wchar_t str[] = L"D:\\Anaconda3\\envs\\tensorflow_vision";

    Py_SetPythonHome(str);

    Py_Initialize();
    np::initialize();
}

public void Close()
{

}

public void Split(mat_cv image, int sliceSize)
{

}

public void Tile(mat_cv[] images, int width, int height)
{

}

np::ndarray ConvertMatToNDArray(const cv::Mat& mat) {
    py::tuple shape = py::make_tuple(mat.rows, mat.cols, mat.channels());
    py::tuple stride = py::make_tuple(mat.channels() * mat.cols * sizeof(uchar), mat.channels() * sizeof(uchar), sizeof(uchar));
    np::dtype dt = np::dtype::get_builtin<uchar>();
    np::ndarray ndImg = np::from_data(mat.data, dt, shape, stride, py::object());

    return ndImg;
}

cv::Mat ConvertNDArrayToMat(const np::ndarray& ndarr) {
    //int length = ndarr.get_nd(); // get_nd() returns num of dimensions. this is used as a length, but we don't need to use in this case. because we know that image has 3 dimensions.
    const Py_intptr_t* shape = ndarr.get_shape(); // get_shape() returns Py_intptr_t* which we can get the size of n-th dimension of the ndarray.
    char* dtype_str = py::extract<char*>(py::str(ndarr.get_dtype()));

    // variables for creating Mat object
    int rows = shape[0];
    int cols = shape[1];
    int channel = shape[2];
    int depth;

    // you should find proper type for c++. in this case we use 'CV_8UC3' image, so we need to create 'uchar' type Mat.
    if (!strcmp(dtype_str, "uint8")) {
        depth = CV_8U;
    }
    else {
        std::cout << "wrong dtype error" << std::endl;
        return cv::Mat();
    }

    int type = CV_MAKETYPE(depth, channel); // CV_8UC3

    cv::Mat mat = cv::Mat(rows, cols, type);
    memcpy(mat.data, ndarr.get_data(), sizeof(uchar) * rows * cols * channel);

    return mat;
}