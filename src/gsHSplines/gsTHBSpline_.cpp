#include <gsCore/gsTemplateTools.h>

#include <gsHSplines/gsTHBSplineBasis.h>
#include <gsHSplines/gsTHBSplineBasis.hpp>

#include <gsHSplines/gsTHBSpline.h>
#include <gsHSplines/gsTHBSpline.hpp>

#include <pybind11/numpy.h>
#include <pybind11/stl.h>


namespace gismo
{

CLASS_TEMPLATE_INST gsTHBSplineBasis <1, real_t>;
CLASS_TEMPLATE_INST gsTHBSplineBasis <2,real_t>;
CLASS_TEMPLATE_INST gsTHBSplineBasis <3,real_t>;
CLASS_TEMPLATE_INST gsTHBSplineBasis <4,real_t>;

CLASS_TEMPLATE_INST gsTHBSpline      <1,real_t>;
CLASS_TEMPLATE_INST gsTHBSpline      <2,real_t>;
CLASS_TEMPLATE_INST gsTHBSpline      <3,real_t>;
CLASS_TEMPLATE_INST gsTHBSpline      <4,real_t>;

namespace internal
{
CLASS_TEMPLATE_INST gsXml< gsTHBSplineBasis<1,real_t> >;
CLASS_TEMPLATE_INST gsXml< gsTHBSplineBasis<2,real_t> >;
CLASS_TEMPLATE_INST gsXml< gsTHBSplineBasis<3,real_t> >;
CLASS_TEMPLATE_INST gsXml< gsTHBSplineBasis<4,real_t> >;

CLASS_TEMPLATE_INST gsXml< gsTHBSpline<1,real_t> >;
CLASS_TEMPLATE_INST gsXml< gsTHBSpline<2,real_t> >;
CLASS_TEMPLATE_INST gsXml< gsTHBSpline<3,real_t> >;
CLASS_TEMPLATE_INST gsXml< gsTHBSpline<4,real_t> >;
}

#ifdef GISMO_WITH_PYBIND11

template <typename Sequence>
inline py::array_t<typename Sequence::value_type> as_pyarray(Sequence&& seq) {
    auto size = seq.size();
    auto data = seq.data();
    std::unique_ptr<Sequence> seq_ptr =
        std::make_unique<Sequence>(std::move(seq));
    auto capsule = py::capsule(seq_ptr.get(), [](void* p) {
        std::unique_ptr<Sequence>(reinterpret_cast<Sequence*>(p));
        });
    seq_ptr.release();
    return py::array(size, data, capsule);
}

template<class T>
py::array_t<T> vector_as_array_nocopy(std::vector<T> vec) {
    return as_pyarray(std::move(vec));
}


template<short_t d, class T>
py::array_t<index_t > get_init_boxes(gsHTensorBasis<d, T> const& basis)
{
    std::vector<index_t > out;

    for (auto lIter = basis.tree().beginLeafIterator(); lIter.good(); lIter.next())
    {
        if (lIter->level > 0)
        {
            out.push_back(lIter->level);
            for (unsigned i = 0; i < d; i++)
            {
                out.push_back(lIter.lowerCorner()[i]);
            }
            for (unsigned i = 0; i < d; i++)
            {
                out.push_back(lIter.upperCorner()[i]);
            }
        }
    }

    /*
    // Write box history (deprecated)
    typename Object::boxHistory const & boxes = obj.get_inserted_boxes();

    for(unsigned int i = 0; i < boxes.size(); i++)
    {
    box.leftCols(d)  = boxes[i].lower.transpose();
    box.rightCols(d) = boxes[i].upper.transpose();

    tmp = putMatrixToXml( box, data, "box" );
    tmp->append_attribute( makeAttribute("level", to_string(boxes[i].level), data ) );
    tp_node->append_node(tmp);
    }
    */

    // All set, return the basis
    return vector_as_array_nocopy(out);
}




namespace py = pybind11;

void pybind11_init_gsTHBSplineBasis2(py::module& m)
{
    using Base = gsBasis<real_t>;
    using Class = gsHTensorBasis<2, real_t>;
    using DClass = gsTHBSplineBasis<2, real_t>;
    py::class_<Class, Base>(m, "gsHTensorBasis2")
        // Member functions
        .def("domainDim", &Class::domainDim,
            "Returns the domain dimension")

        .def("numActive", &Class::numActive,
            "Returns the number of active (nonzero) functions")
        .def("numActive_into", &Class::numActive_into,
            "Returns the number of active (nonzero) functions into a matrix")

        .def("anchors", &Class::anchors,
            "Returns the anchor points that represent the members of the basis")
        .def("active_into", &Class::anchors_into,
            "Returns the anchor points that represent the members of the basis into a matrix")



        .def("active", &Class::active,
            "Returns the indices of active (nonzero) functions")
        .def("active_into", &Class::active_into,
            "Returns the indices of active (nonzero) functions into a matrix")

        .def("eval_into", &Class::eval_into,
            "Evaluates the values into a matrix")
        .def("deriv_into", &Class::deriv_into,
            "Evaluates the derivatives into a matrix")
        .def("deriv2_into", &Class::deriv2_into,
            "Evaluates the second derivatives into a matrix")
        .def("evalSingle_into", &Class::evalSingle_into,
            "Evaluates the values of a single basis function into a matrix")
        .def("derivSingle_into", &Class::derivSingle_into,
            "Evaluates the derivatives of a single basis functioninto a matrix")
        .def("deriv2Single_into", &Class::deriv2Single_into,
            "Evaluates the second derivatives of a single basis function into a matrix")

        // Derived from gsHTensorBasis
        .def("size", &Class::size, "Returns the domain dimension")
        .def("support", static_cast<gsMatrix<real_t>(Class::*)() const> (&Class::support), "Get the support of the basis")
        .def("support", static_cast<gsMatrix<real_t>(Class::*)(const index_t&) const> (&Class::support), "Get the support of the basis function with an index i")
        .def("tensorLevel", &Class::tensorLevel, "Returns the tensor basis on level i")
        .def("uniformRefine", static_cast<void (Class::*)(int, int, int)> (&Class::uniformRefine), "Refines the basis uniformly",
            py::arg("numKnots") = 1, py::arg("mul") = 1, py::arg("dir") = -1) //default arguments

        .def("refine", static_cast<void (Class::*)(gsMatrix<real_t> const&, int)> (&Class::refine), "Refines the basis given a box")
        .def("unrefine", static_cast<void (Class::*)(gsMatrix<real_t> const&, int)> (&Class::unrefine), "Refines the basis given a box")
        .def("refine", static_cast<void (Class::*)(gsMatrix<real_t> const&)> (&Class::refine), "Refines the basis given a box")
        // .def("unrefine", static_cast<void (Class::*)(gsMatrix<real_t> const &     )> (&Class::unrefine), "Refines the basis given a box")
        .def("refineElements", &Class::refineElements, "Refines the basis given elements  ")
        .def("unrefineElements", &Class::unrefineElements, "Unrefines the basis given elements")
        .def("asElements", &Class::asElements, "Returns the elements given refinement boxes")
        .def("asElementsUnrefine", &Class::asElementsUnrefine, "Returns the elements given refinement boxes")


        .def("_get_init_boxes", &get_init_boxes<2, real_t>, "get the boxes vector which can be passed to the constructor")
        ;
    py::class_<DClass, Class>(m, "gsTHBSplineBasis2")

      // Constructors
      .def(py::init<>())
      .def(py::init < gsTensorBSplineBasis<2, real_t> const&>())
      .def(py::init<gsTensorBSplineBasis<2, real_t> const&, std::vector<index_t>&>())
      .def(py::init<gsTensorBSplineBasis<2, real_t> const&, gsMatrix<real_t> const&>())
      .def(py::init<gsTensorBSplineBasis<2, real_t> const&, gsMatrix<real_t> const&, std::vector<index_t> const &>())

    //.def(py::init<gsBasis<real_t> const&                                         >())
      ;

}

void pybind11_init_gsTHBSplineBasis3(py::module &m)
{
  using Base  = gsBasis<real_t>;
  using Class = gsTHBSplineBasis<3,real_t>;
  py::class_<Class,Base>(m, "gsTHBSplineBasis3")

    // Constructors
    .def(py::init<>())
    .def(py::init<gsTensorBSplineBasis<3,real_t> const&, std::vector<index_t>   &>())
    .def(py::init<gsTensorBSplineBasis<3,real_t> const&, gsMatrix<real_t> const &>())
    .def(py::init<gsBasis<real_t> const&                                         >())

    // Member functions
    .def("domainDim", 			&Class::domainDim,
    		"Returns the domain dimension"												)
    .def("eval_into", 			&Class::eval_into,
    		"Evaluates the values into a matrix"										)
    .def("deriv_into", 			&Class::deriv_into,
    		"Evaluates the derivatives into a matrix"									)
    .def("deriv2_into", 		&Class::deriv2_into,
    		"Evaluates the second derivatives into a matrix"							)
    .def("evalSingle_into", 	&Class::evalSingle_into,
    		"Evaluates the values of a single basis function into a matrix"				)
    .def("derivSingle_into", 	&Class::derivSingle_into,
    		"Evaluates the derivatives of a single basis functioninto a matrix" 		)
    .def("deriv2Single_into", 	&Class::deriv2Single_into,
    		"Evaluates the second derivatives of a single basis function into a matrix" )
    ;
}

void pybind11_init_gsTHBSplineBasis4(py::module &m)
{
  using Base  = gsBasis<real_t>;
  using Class = gsTHBSplineBasis<4,real_t>;
  py::class_<Class,Base>(m, "gsTHBSplineBasis4")

    // Constructors
    .def(py::init<>())
    .def(py::init<gsTensorBSplineBasis<4,real_t> const&, std::vector<index_t>   &>())
    .def(py::init<gsTensorBSplineBasis<4,real_t> const&, gsMatrix<real_t> const &>())
    .def(py::init<gsBasis<real_t> const&                                         >())

    // Member functions
    .def("domainDim", 			&Class::domainDim,
    		"Returns the domain dimension"												)
    .def("eval_into", 			&Class::eval_into,
    		"Evaluates the values into a matrix"										)
    .def("deriv_into", 			&Class::deriv_into,
    		"Evaluates the derivatives into a matrix"									)
    .def("deriv2_into", 		&Class::deriv2_into,
    		"Evaluates the second derivatives into a matrix"							)
    .def("evalSingle_into", 	&Class::evalSingle_into,
    		"Evaluates the values of a single basis function into a matrix"				)
    .def("derivSingle_into", 	&Class::derivSingle_into,
    		"Evaluates the derivatives of a single basis functioninto a matrix" 		)
    .def("deriv2Single_into", 	&Class::deriv2Single_into,
    		"Evaluates the second derivatives of a single basis function into a matrix" )
    ;
}

void pybind11_init_gsTHBSpline2(py::module &m)
{
  using Base  = gsGeometry<real_t>;
	using Class = gsTHBSpline<2,real_t>;
	py::class_<Class,Base>(m, "gsTHBSpline2")

	// Constructors
	.def(py::init<>())
	// this one does not work:
	// .def(py::init<const gsTHBSplineBasis<2,real_t> *, const gsMatrix<real_t> * >())
	.def(py::init<const gsTHBSplineBasis<2,real_t> &, const gsMatrix<real_t> & >())
	.def(py::init<const gsTensorBSpline<2,real_t> &                      >())

	// Member functions
	.def("domainDim", 			&Class::domainDim,
			"Returns the domain dimension"					)
	.def("eval_into", 			&Class::eval_into,
			"Evaluates the values into a matrix"			)
	.def("deriv_into", 			&Class::deriv_into,
			"Evaluates the derivatives into a matrix"		)
	.def("deriv2_into", 		&Class::deriv2_into,
			"Evaluates the second derivatives into a matrix")
	;
}

void pybind11_init_gsTHBSpline3(py::module &m)
{
  using Base  = gsGeometry<real_t>;
	using Class = gsTHBSpline<3,real_t>;
	py::class_<Class,Base>(m, "gsTHBSpline3")

	// Constructors
	.def(py::init<>())
	// this one does not work:
	// .def(py::init<const gsTHBSplineBasis<3,real_t> *, const gsMatrix<real_t> * >())
	.def(py::init<const gsTHBSplineBasis<3,real_t> &, const gsMatrix<real_t> & >())
	.def(py::init<const gsTensorBSpline<3,real_t> &                      >())

	// Member functions
	.def("domainDim", 			&Class::domainDim,
			"Returns the domain dimension"					)
	.def("eval_into", 			&Class::eval_into,
			"Evaluates the values into a matrix"			)
	.def("deriv_into", 			&Class::deriv_into,
			"Evaluates the derivatives into a matrix"		)
	.def("deriv2_into", 		&Class::deriv2_into,
			"Evaluates the second derivatives into a matrix")
	;
}

void pybind11_init_gsTHBSpline4(py::module &m)
{
  using Base  = gsGeometry<real_t>;
	using Class = gsTHBSpline<4,real_t>;
	py::class_<Class,Base>(m, "gsTHBSpline4")

	// Constructors
	.def(py::init<>())
	// this one does not work:
	// .def(py::init<const gsTHBSplineBasis<4,real_t> *, const gsMatrix<real_t> * >())
	.def(py::init<const gsTHBSplineBasis<4,real_t> &, const gsMatrix<real_t> & >())
	.def(py::init<const gsTensorBSpline<4,real_t> &                      >())

	// Member functions
	.def("domainDim", 			&Class::domainDim,
			"Returns the domain dimension"					)
	.def("eval_into", 			&Class::eval_into,
			"Evaluates the values into a matrix"			)
	.def("deriv_into", 			&Class::deriv_into,
			"Evaluates the derivatives into a matrix"		)
	.def("deriv2_into", 		&Class::deriv2_into,
			"Evaluates the second derivatives into a matrix")
	;
}

#endif

}
