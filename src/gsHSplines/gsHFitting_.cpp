#include <gsCore/gsTemplateTools.h>

#include <gsModeling/gsFitting.h>
#include <gsHSplines/gsHFitting.h>
#include <gsHSplines/gsHTensorBasis.h>

#include <gsNurbs/gsBSplineBasis.h>

namespace gismo
{

CLASS_TEMPLATE_INST gsHFitting<2, real_t>;


#ifdef GISMO_WITH_PYBIND11


namespace py = pybind11;
void pybind11_init_gsHFitting2(py::module &m)
{
    using Base = gsFitting<real_t>;
    using Class = gsHFitting<2, real_t>;
  //py::class_<Class, Base>(m, "gsHFitting2")
      py::class_<Class>(m, "gsHFitting2")


    // Constructors
   // .def( py::init<>() ) // Empty constructor
    .def( py::init<gsMatrix<real_t> const &, gsMatrix<real_t> const &, gsHTensorBasis<2, real_t>&, real_t, const std::vector< unsigned >&>() )
    .def( py::init<gsMatrix<real_t> const &, gsMatrix<real_t> const &, gsHTensorBasis<2, real_t>&, real_t, const std::vector< unsigned >& , real_t >() )

    // Member functions
    .def("maxPointError", &Class::maxPointError, "Returns the maximum point-wise error from the pount cloud (or zero if not fitted)")
    .def("minPointError", &Class::minPointError, "Returns the minimum point-wise error from the pount cloud (or zero if not fitted)")
    .def("numPointsBelow", &Class::numPointsBelow, "Computes the number of points below the error threshold (or zero if not fitted)")
    .def("nextIteration", static_cast<bool (Class::*)(real_t, real_t, index_t)> (&Class::nextIteration), "One step of the refinement of iterative_refine(...)")
    .def("result", &Class::result, "gives back the computed approximation", py::return_value_policy::reference_internal)

    .def("compute", &Class::compute, "Computes the least square fit for a gsBasis.")
    .def("applySmoothing", &Class::applySmoothing, "apply smoothing to the input matrix.")
    .def("smoothingMatrix", &Class::smoothingMatrix, "get the amoothing matrix.")
    .def("parameterCorrection", &Class::parameterCorrection, "Apply parameter correction steps.")
    .def("pointWiseErrors", &Class::pointWiseErrors, "Return the errors for each point.")
    ;
}
#endif


} // namespace gismo
