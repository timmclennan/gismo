
#include <gsHSplines/gsHFitting.h>
#include <gsHSplines/gsHTensorBasis.h>
#include <gsHSplines/gsTHBSpline.h>
#include <gsHSplines/gsHBSpline.h>

#include <gsHSplines/py_gsHSplines.h>
#include <pybind11/stl_bind.h>

namespace gismo
{

#ifdef GISMO_WITH_PYBIND11

namespace py = pybind11;

void pybind11_init_gsHsplines(py::module &m)
{
    py::module hsplines = m.def_submodule("hsplines");

    hsplines.attr("__name__") = "pygismo.hsplines";
    hsplines.attr("__version__") = GISMO_VERSION;
    hsplines.doc() = "G+Smo (Geometry + Simulation Modules): HSplines module";
//    py::bind_vector<std::vector<index_t>>(hsplines, "IndexVector", py::buffer_protocol());

    gismo::pybind11_init_gsHBSplineBasis2(hsplines);
    gismo::pybind11_init_gsHBSplineBasis3(hsplines);
    gismo::pybind11_init_gsHBSplineBasis4(hsplines);
    gismo::pybind11_init_gsHBSpline2(hsplines);
    gismo::pybind11_init_gsHBSpline3(hsplines);
    gismo::pybind11_init_gsHBSpline4(hsplines);
    gismo::pybind11_init_gsTHBSplineBasis2(hsplines);
    gismo::pybind11_init_gsTHBSplineBasis3(hsplines);
    gismo::pybind11_init_gsTHBSplineBasis4(hsplines);
    gismo::pybind11_init_gsTHBSpline2(hsplines);
    gismo::pybind11_init_gsTHBSpline3(hsplines);
    gismo::pybind11_init_gsTHBSpline4(hsplines);
    gismo::pybind11_init_gsHFitting2(hsplines);

    ;
}
#endif


} // namespace gismo
