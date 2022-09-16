/** @file gsLowRankFitting.h

    @brief Fitting gridded data with low-rank B-splines.

    This file is part of the G+Smo library.
    
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.

    Author(s): D. Mokris
*/

#pragma once

#include <gsCore/gsForwardDeclarations.h>
#include <gsCore/gsLinearAlgebra.h>
#include <gsMatrix/gsEigenDeclarations.h>
#include <gsUtils/gsStopwatch.h>
#include <gsIO/gsWriteParaview.h>
#include <gsNurbs/gsBSpline.h>

#include <gsModeling/gsFitting.h>
#include <gsMatrix/gsSvd.h>
#include <gsMatrix/gsMatrixUtils.h>
#include <gsMatrix/gsMatrixCrossApproximation.h>
#include <gsMatrix/gsMatrixCrossApproximation_3.h>
#include <gsModeling/gsL2Error.h>
#include <gsIO/gsWriteGnuplot.h>

#include <random>

namespace gismo
{

    enum class gsErrType {l2, max, L2};

template <class T>
class gsLowRankFitting : public gsFitting<T>
{
public:

    // for param
    gsLowRankFitting()
    {
    }

    gsLowRankFitting(const gsMatrix<T>& params,
		     const gsMatrix<T>& points,
		     const gsVector<T>& uWeights,
		     const gsVector<T>& vWeights,
		     gsTensorBSplineBasis<2, T>& basis,
		     T zero = 1e-13,
		     index_t sample = -1,
		     gsErrType errType = gsErrType::l2)
	: gsFitting<T>(params, points, basis), m_zero(zero), m_sample(sample), m_errType(errType)
    {
	initPQ(matrixUtils::diag(uWeights),
	       matrixUtils::diag(vWeights));
    }

    /**
       Constructor.

       \param uNpts Number of data points in u-direction. Special
       value -1 means that params form a quadratic array and the value
       can be inferred.
     */
    gsLowRankFitting(const gsMatrix<T>& params,
		     const gsMatrix<T>& points,
		     gsTensorBSplineBasis<2, T>& basis,
		     T zero = 1e-13,
		     index_t sample = -1,
		     gsErrType errType = gsErrType::l2,
		     index_t uNpts = -1)
	: gsFitting<T>(params, points, basis), m_zero(zero), m_sample(sample), m_errType(errType), m_uNpts(uNpts)
    {
	// Note: Forgetting the <T> leads to a confusing error message.

	// If uNpts not given, compute it.
	if(m_uNpts == -1)
	    m_uNpts = math::sqrt(params.cols());

	index_t vNpts = params.cols() / m_uNpts;
	initPQ(matrixUtils::identity<T>(m_uNpts),
	       matrixUtils::identity<T>(  vNpts));
    }

    void computeCross(bool pivot, index_t maxIter, index_t sample);

    void computeCross_3(bool pivot, index_t maxIter, index_t sample);
    
    void computeSVD(index_t maxIter, index_t sample, const std::string& filename);

    void computeRes();

    // Method B with weights.
    void computeFull(const gsVector<T>& uWeights, const gsVector<T>& vWeights);

    // Method B with more details.
    void computeFull()
    {
	index_t uNpts = math::sqrt(this->m_param_values.cols());
	gsVector<T> identity = matrixUtils::oneVector<T>(uNpts);
	computeFull(identity, identity);
    }

    void computeCrossWithRef(bool pivot, index_t maxIter, index_t sample);

    void computeCrossWithRefAndStop(T tol, bool pivot);

    int computeCrossWithStop(T epsAccept, T epsAbort, bool pivot = true);

    T methodB(bool printErr);

    T methodC(bool printErr, index_t maxIter);

    void CR2I_old(const gsMatrix<T>& bott,
		  const gsMatrix<T>& left,
		  const gsMatrix<T>& rght,
		  const gsMatrix<T>& topp) const;

    void CR2I_new(const gsMatrix<T>& bott,
		  const gsMatrix<T>& left,
		  const gsMatrix<T>& rght,
		  const gsMatrix<T>& topp) const;

    //T L2error() const;

    inline void exportl2Err(const std::string& filename) const
    {
	gsWriteGnuplot(m_l2Err, filename);
    }

    inline void exportL2Err(const std::string& filename) const
    {
	// We use getL2Err to cover the case of L2-errors not initialized.
	gsWriteGnuplot(getL2Err(), filename);
    }

    inline void exportMaxErr(const std::string& filename) const
    {
	gsWriteGnuplot(m_maxErr, filename);
    }

    inline void exportDecompErr(const std::string& filename) const
    {
	gsWriteGnuplot(m_decompErr, filename);
    }

    inline const std::vector<T>& getl2Err() const
    {
	return m_l2Err;
    }

    inline const std::vector<T> getL2Err() const
    {
	if(m_L2Err.empty())
	{
	    gsWarn << "L2-error not computed, returning -1s." << std::endl;
	    return std::vector<T>(m_l2Err.size(), T(-1));
	}
	else
	    return m_L2Err;
    }

    inline const std::vector<T>& getMaxErr() const
    {
	return m_maxErr;
    }

    void testMN(index_t rows) const
    {
	gsMatrix<T> test = convertToMN(rows);
	gsFileData<T> fd;
	fd << test;
	fd.dump("test");
    }

    index_t getRank() const
    {
	return m_rank;
    }

protected:

    index_t partitionParam(gsMatrix<T>& uPar, gsMatrix<T>& vPar) const;

    gsMatrix<T> convertToMN(index_t rows) const;

    gsSparseMatrix<T> convertToSparseMN(index_t rows) const;

    gsMatrix<T> convertBack(const gsMatrix<T>& points) const;

    gsMatrix<T> getErrorsMN(size_t rows) const;

    void initPQ(const gsMatrix<T>& uWeights, const gsMatrix<T>& vWeights);

protected:

    gsMatrix<T> m_X, m_Y; // Transposed(!) collocation matrices.

    gsMatrix<T> m_P, m_Q; // The matrices P and Q from the paper.

    std::vector<T> m_l2Err, m_L2Err, m_maxErr, m_decompErr;

    index_t m_rank;

    /// What is considered zero during the ACA computation.
    T m_zero;

    /// Which analytical solution to compare against in the L^2-norm.
    /// Special value -1 means do not compute the L^2-norm at all.
    index_t m_sample;

    gsErrType m_errType;

    // Number of data points in u-direction.
    index_t m_uNpts;
};

} // namespace gismo
