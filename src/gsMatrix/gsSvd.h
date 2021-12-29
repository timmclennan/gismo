/** @file gsSvd.h

    @brief A little wrapper around the SVD.

    This file is part of the G+Smo library.
    
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.

    Author(s): D. Mokris
*/

#pragma once

#include <gsMatrix/gsVector.h>
#include <gsMatrix/gsMatrix.h>
#include <gsMatrix/gsMatrixUtils.h>

namespace gismo
{

template <class T>
class gsSvd
{
public:
    gsSvd(const gsMatrix<T>& mat)
    {
	Eigen::JacobiSVD< Eigen::Matrix<real_t, Dynamic, Dynamic> > svd(mat, Eigen::ComputeFullU | Eigen::ComputeFullV);
	m_U = svd.matrixU();
	m_S = svd.singularValues();
	m_V = svd.matrixV();
	m_rank = svd.rank();

	// gsInfo << "U:\n" << m_U << std::endl;
	// gsInfo << "S:\n" << m_S << std::endl;
	// gsInfo << "V:\n" << m_V << std::endl;

	// gsMatrix<T> S_mat(3, 3);
	// S_mat.setZero();
	// S_mat(0, 0) = m_S(0);
	// S_mat(1, 1) = m_S(1);
	// S_mat(2, 2) = m_S(2);

	// gsInfo << "Product:\n" << m_U * S_mat * m_V.transpose() << std::endl;
    }

    gsVector<T> u(index_t i) const
    {
	GISMO_ASSERT(0 <= i && i < m_U.cols(), "i out of U-bounds");
	return m_U.col(i);
    }

    T s(index_t i) const
    {
	GISMO_ASSERT(0 <= i && i < m_S.rows(), "i out of S-bounds");
	return m_S(i, 0);
    }

    gsVector<T> v(index_t j) const
    {
	// We take cols and not rows, because A = U * S * V^T and not V.
	GISMO_ASSERT(0 <= j && j < m_V.cols(), "j out of V-bounds");
	return m_V.col(j);
    }

    index_t rank() const
    {
	return m_rank;
    }

    bool sanityCheck(const gsMatrix<T> test)
    {
	if(m_U.rows() != test.rows())
	{
	    gsInfo << m_U.rows() << " rows of U, expected " << test.rows();
	    return false;
	}

	if(m_V.cols() != test.cols())
	{
	    gsInfo << m_V.cols() << " cols of V, expected " << test.cols();
	    return false;
	}
		
	gsMatrix<T> result(m_U.rows(), m_V.cols());
	result.setZero();

	for(index_t i=0; i<m_S.rows(); i++)
	{
	    result += s(i) * matrixUtils::tensorProduct(u(i), v(i));
	}
	gsInfo << "Input:\n" << test << "\nCheck:\n" << result << std::endl;

	for(index_t i=0; i<result.rows(); i++)
	{
	    for(index_t j=0; j<result.cols(); j++)
		if( test(i, j) != result(i, j))
		{
		    gsInfo << "Different values on position " << i << ", " << j << "; got " << result(i, j) << ", expected " << test(i, j) << std::endl;
		    return false;
		}
	}
	return true;
    }

 protected:
    gsMatrix<T> m_U, m_S, m_V;
    index_t m_rank;
};

} // namespace gismo
