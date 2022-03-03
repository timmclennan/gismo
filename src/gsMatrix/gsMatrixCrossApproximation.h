/** @file gsMatrixCrossApproximation.h

    @brief Greedy algorithm for approximating matrices with tensor-products.

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

/** This is an algorithm similar to that described in Section 2.3 of
I. Georgieva, C. Hofreither: An algorithm for low-rank approximation
of bivariate functions using splines, JCAM 310, pp. 80 -- 91, 2017.

However, in addition to row pivoting there is also a possibility to
search for a maximum in the entire matrix. That is more precise but
probably much slower.
*/

template <class T>
class gsMatrixCrossApproximation
{
public:
    gsMatrixCrossApproximation(const gsMatrix<T>& matrix)
	: m_mat(matrix), m_ik(0)
    {
    }

    bool nextIteration(T& sigma, gsVector<T>& uVec, gsVector<T>& vVec, bool pivot)
    {
	if(pivot)
	    return nextPivotingIteration(sigma, uVec, vVec);
	else
	    return nextFullIteration(sigma, uVec, vVec);
    }

    void updateMatrix(const gsMatrix<T>& matrix)
    {
	if(m_mat.rows() != matrix.rows() || m_mat.cols() != matrix.cols())
	    gsWarn << "Warning: replacing a ("
		   << m_mat.rows()  << " x " << m_mat.cols() << ") matrix with a ("
		   << matrix.rows() << " x " << matrix.cols() << ") one." << std::endl;

	m_mat = matrix;
    }

protected:

    bool nextFullIteration(T& sigma, gsVector<T>& uVec, gsVector<T>& vVec)
    {
	index_t i=0, j=0;
	if( !findAbsMax(i, j))
	    return false;

	// Note to future self: the order in the tensor-product is
	// somewhat surprising, u is a column and v a row.
	sigma = 1.0 / m_mat(i, j);
	uVec  = m_mat.col(j);
	vVec  = m_mat.row(i);

	matrixUtils::addTensorProduct(m_mat, -1 * sigma, uVec, vVec);
	return true;
    }

    bool nextPivotingIteration(T& sigma, gsVector<T>& uVec, gsVector<T>& vVec, index_t shifts = 0)
    {
	if(shifts > m_mat.rows())
	{
	    // This way we prevent a possible infinite loop.
	    //gsInfo << "Shifted too much." << std::endl;
	    return false;
	}

	index_t jk = findAbsMaxRow(m_ik);
	if(m_mat(m_ik, jk) == 0)
	{
	    //gsInfo << "shifting" << std::endl;
	    m_ik = (m_ik + 1) % m_mat.rows();
	    return nextPivotingIteration(sigma, uVec, vVec, shifts+1);
	}
	else
	{
	    // gsInfo << "approximating\n" << m_mat << std::endl;
	    // gsInfo << "(i, j) = (" << m_ik << ", " << jk << ")\n";
	    sigma = 1.0 / m_mat(m_ik, jk);
	    uVec  = m_mat.col(jk);
	    vVec  = m_mat.row(m_ik);

	    m_ik = findAbsMaxCol(jk);
	    matrixUtils::addTensorProduct(m_mat, -1 * sigma, uVec, vVec);
	    return true;
	}
    }

    /*index_t findAbsMaxRow(index_t i) const
    {
	index_t jk = 0;
	T max = 0;
	for(index_t j=0; j<m_mat.cols(); j++)
	{
	    T curr = math::abs(m_mat(i, j));
	    if(curr > max)
	    {
		max = curr;
		jk = j;
	    }
	}
	return jk;
	}*/

    index_t findAbsMaxRow(index_t i) const
    {
	index_t* result = new index_t(0);
	m_mat.row(i).maxCoeff(result);
	return *result;
    }

    /*index_t findAbsMaxCol(index_t j) const
    {
	index_t ik = 0;
	T max = 0;
	for(index_t i=0; i<m_mat.rows(); i++)
	{
	    T curr = math::abs(m_mat(i, j));
	    if(curr > max)
	    {
		max = curr;
		ik = i;
	    }
	}
	return ik;
	}*/
    index_t findAbsMaxCol(index_t j) const
    {
	index_t* result = new index_t(0);
	m_mat.col(j).maxCoeff(result);
	return *result;
    }

    /// Finds the element with the highest absolute value
    /// and returns false iff the max is equal to 0.
    bool findAbsMax(index_t& i_res, index_t& j_res) const
    {
	i_res = 0;
	j_res = 0;
	T max = 0;
	for(index_t i=0; i<m_mat.rows(); i++)
	{
	    for(index_t j=0; j<m_mat.cols(); j++)
	    {
		T curr = math::abs(m_mat(i,j));
		if(curr > max)
		{
		    max = curr;
		    i_res = i;
		    j_res = j;
		}
	    }
	}

	return (math::abs(max) > 1e-15);
    }

protected: // elements

    gsMatrix<T> m_mat;

    index_t m_ik;
    
};

} // namespace gismo
