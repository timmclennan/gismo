/** @file py_gsHsplines.h

    @brief  hierarchical splines python module setup

*/
#include <gsCore/gsExport.h>

#pragma once

namespace gismo {


#ifdef GISMO_WITH_PYBIND11

GISMO_EXPORT
void pybind11_init_gsHsplines(pybind11::module& m);

#endif // GISMO_WITH_PYBIND11

}// namespace gismo

