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
		     gsTensorBSplineBasis<2, T>& basis)
	: gsFitting<T>(params, points, basis)
    {
	initPQ(matrixUtils::diag(uWeights),
	       matrixUtils::diag(vWeights));
    }

    gsLowRankFitting(const gsMatrix<T>& params,
		     const gsMatrix<T>& points,
		     gsTensorBSplineBasis<2, T>& basis)
	: gsFitting<T>(params, points, basis)
    {
	// Note: Forgetting the <T> leads to a confusing error message.
	index_t uNpts = math::sqrt(params.cols());
	initPQ(matrixUtils::identity<T>(uNpts),
	       matrixUtils::identity<T>(uNpts));
    }

    void computeCross(bool pivot, index_t maxIter, index_t sample);

    void computeCross_3(bool pivot, index_t maxIter, index_t sample);
    
    void computeSVD(index_t maxIter, index_t sample, const std::string& filename);

    void computeRes();

    void computeCrossWithRef(bool pivot, index_t maxIter, index_t sample);

    void computeCrossWithRefAndStop(T tol, bool pivot);

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

    inline void exportL2Err(const std::string& filename) const
    {
	gsWriteGnuplot(m_L2Err, filename);
    }

    inline void exportMaxErr(const std::string& filename) const
    {
	gsWriteGnuplot(m_MaxErr, filename);
    }

    inline const std::vector<T>& getL2Err() const
    {
	return m_L2Err;
    }

    inline const std::vector<T>& getMaxErr() const
    {
	return m_MaxErr;
    }

    void testMN(index_t rows) const
    {
	gsMatrix<T> test = convertToMN(rows);
	gsFileData<T> fd;
	fd << test;
	fd.dump("test");
    }

protected:

    index_t partitionParam(gsMatrix<T>& uPar, gsMatrix<T>& vPar) const;

    gsMatrix<T> convertToMN(index_t rows) const;

    gsSparseMatrix<T> convertToSparseMN(index_t rows) const;

    gsMatrix<T> convertBack(const gsMatrix<T>& points) const;

    gsMatrix<T> getErrorsMN(size_t rows) const;

    void initPQ(const gsMatrix<T>& uWeights, const gsMatrix<T>& vWeights);

protected:

    gsMatrix<T> m_uWeights, m_vWeights; // TODO: remove!

    gsMatrix<T> m_X, m_Y; // Transposed(!) collocation matrices.

    gsMatrix<T> m_P, m_Q; // The matrices P and Q from the paper.

    std::vector<T> m_L2Err, m_MaxErr;
};

} // namespace gismo
