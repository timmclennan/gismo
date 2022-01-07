/** @file gsMatrixUtils.h

    @brief A few util functions for matrix manipulations.

    This file is part of the G+Smo library.
    
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.

    Author(s): D. Mokris
*/

#pragma once

#include <gsMatrix/gsVector.h>
#include <gsMatrix/gsMatrix.h>

namespace gismo
{
    namespace matrixUtils
    {
	template <class T>
	gsMatrix<T> tensorProduct(const gsVector<T>& u, const gsVector<T>& v)
	{
	    gsMatrix<T> result(u.size(), v.size());
	    for(index_t i=0; i<u.size(); i++)
		for(index_t j=0; j<v.size(); j++)
		    result(i, j) = u(i) * v(j);

	    return result;
	}

	template <class T>
	void addTensorProduct(gsMatrix<T>& result, T sigma, const gsVector<T>& u, const gsVector<T>& v)
	{
	    GISMO_ASSERT(u.size() == result.rows(), "row mismatch in addTensorProduct");
	    GISMO_ASSERT(v.size() == result.cols(), "col mismatch in addTensorProduct");

	    for(index_t i=0; i<result.rows(); i++)
		for(index_t j=0; j<result.cols(); j++)
		    result(i, j) += sigma * u(i) * v(j);
	}

	template <class T>
	T det2x2(T c00, T c0n, T cm0, T cmn)
	{
	    // gsMatrix<T> mat(2, 2);
	    // mat << c00, c0n,
	    // 	cm0, cmn;
	    // return mat.det();
	    return c00 * cmn - c0n * cm0;
	}

    } // namespace matrixUtils

} // namespace gismo
