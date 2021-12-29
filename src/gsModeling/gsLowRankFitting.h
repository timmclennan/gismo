/** @file gsLowRankFitting.h

    @brief Fitting gridded data with low-rank B-splines.

    This file is part of the G+Smo library.
    
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.

    Author(s): D. Mokris
*/

#pragma once

#include <gsModeling/gsFitting.h>
#include <gsMatrix/gsSvd.h>
#include <gsMatrix/gsMatrixUtils.h>
#include <gsMatrix/gsMatrixCrossApproximation.h>

namespace gismo
{

template <class T>
class gsLowRankFitting : public gsFitting<T>
{
public:

    gsLowRankFitting(const gsMatrix<T>& params,
		     const gsMatrix<T>& points,
		     gsTensorBSplineBasis<2, T>& basis)
	: gsFitting<T>(params, points, basis)
    {
	// Note: Forgetting the <T> leads to a confusing error message.
    }

    void computeCross(bool pivot);
    
    void computeSVD();

    void computeRes();

protected:

    index_t partitionParam(gsMatrix<T>& uPar, gsMatrix<T>& vPar) const;

    gsMatrix<T> convertToMN(index_t rows) const;

    gsMatrix<T> convertBack(const gsMatrix<T>& points) const;

    gsMatrix<T> getErrorsMN(size_t rows) const;
};

template <class T>
index_t gsLowRankFitting<T>::partitionParam(gsMatrix<T>& uPar, gsMatrix<T>& vPar) const
{
    index_t uNum = 0;
    for(index_t i=0; this->m_param_values(0, i) != 1; i++)
	uNum++;
    uNum++; // For the last iteration.
    //gsInfo << "uNum: " << uNum << std::endl;

    // Separate the u- and v-parameters.
    // TODO: Rectangular arrays.
    uPar.resize(1, uNum);
    vPar.resize(1, uNum);
    for(index_t i=0; i<uNum; i++)
    {
    	uPar(0, i) = this->m_param_values(0, i);
    	vPar(0, i) = this->m_param_values(1, i * uNum);
    }
    // gsInfo << "uPar:\n" << uPar << std::endl;
    // gsInfo << "vPar:\n" << vPar << std::endl;
    return uNum;
}

template <class T>
gsMatrix<T> gsLowRankFitting<T>::convertToMN(index_t rows) const
{
    gsMatrix<T> result(rows, this->m_points.rows() / rows);
    for(index_t i=0; i<this->m_points.rows(); i++)
	result(i%rows, i/rows) = this->m_points(i, 0);
    return result;
}

template <class T>
gsMatrix<T> gsLowRankFitting<T>::convertBack(const gsMatrix<T>& points) const
{
    gsMatrix<T> result(1, points.rows() * points.cols());
    for(index_t i=0; i<points.rows(); i++)
	for(index_t j=0; j<points.cols(); j++)
	    result(points.cols() * j + i) = points(i, j);

    return result;
	    
}

template <class T>
gsMatrix<T> gsLowRankFitting<T>::getErrorsMN(size_t rows) const
{
    // Challenge: Can we merge this and convertToMN into a single function?

    gsMatrix<T> val_i;
    this->m_result->eval_into(this->m_param_values, val_i);

    gsMatrix<T> result(rows, this->m_pointErrors.size() / rows);
    for(size_t i=0; i<this->m_pointErrors.size(); i++)
	result(i%rows, i/rows) = this->m_points(i, 0) - val_i(0, i);
    // Signed distance, works in 1D only.
	    //this->m_pointErrors[i];

    return result;
}

template <class T>
void gsLowRankFitting<T>::computeSVD()
{
    
    gsBSplineBasis<T> uBasis = *(static_cast<gsBSplineBasis<T>*>(&(this->m_basis->component(0))));
    gsBSplineBasis<T> vBasis = *(static_cast<gsBSplineBasis<T>*>(&(this->m_basis->component(1))));

    gsMatrix<real_t> coefs(uBasis.size(), vBasis.size());
    coefs.setZero();

    // 0. Convert points to an m x n matrix.
    gsMatrix<T> uPar, vPar;
    index_t uNum = partitionParam(uPar, vPar);
    gsMatrix<real_t> ptsMN = convertToMN(uNum);
    //gsMatrix<real_t> check = convertBack(ptsMN);
    // gsInfo << "ptsMN (" << ptsMN.rows() << " x " << ptsMN.cols() << ")" << std::endl;
    // gsInfo << ptsMN << std::endl;
    //gsInfo << "orig (" << points.rows() << " x " << points.cols() << "):\n" << points << std::endl;
    //gsInfo << "check (" << check.rows() << " x " << check.cols() << "):\n" << check << std::endl;

    // 1. Perform an SVD on it.
    gsSvd<T> pointSVD(ptsMN);
    //pointSVD.sanityCheck(ptsMN);
    gsInfo << "Rank: " << pointSVD.rank() << std::endl;

    // 2. Iterate.
    for(index_t r=0; r<pointSVD.rank(); r++)
    {
	// gsInfo << "uPar:\n" << uPar << std::endl;
	// gsInfo << "uPts:\n" << pointSVD.u(r).transpose() << std::endl;

	// 2.0. Fit u with the u-basis.
	gsFitting<real_t> uFitting(uPar, pointSVD.u(r).transpose(), uBasis);
	uFitting.compute();
	// A note to future self: Forgetting about the transpose lead to preconditioners failing.

	// gsInfo << "vPar:\n" << vPar << std::endl;
	// gsInfo << "vPts:\n" << pointSVD.v(r).transpose() << std::endl;

	// 2.1. Fit v with the v-basis.
	gsFitting<real_t> vFitting(vPar, pointSVD.v(r).transpose(), vBasis);
	vFitting.compute();

	// gsWriteParaview(*uFitting.result(), "u-fit", 100, false, true);
	// gsWriteParaview(*vFitting.result(), "v-fit", 100, false, true);

	// 2.3. Add the coefficients to the running tensor-product B-spline.
	gsVector<real_t> uCoefs = uFitting.result()->coefs().col(0);
	gsVector<real_t> vCoefs = vFitting.result()->coefs().col(0);
	coefs += pointSVD.s(r) * matrixUtils::tensorProduct(uCoefs, vCoefs);

	delete this->m_result;
	this->m_result = this->m_basis->makeGeometry(give(convertBack(coefs).transpose())).release();
	this->computeErrors();
	gsInfo << this->maxPointError() << std::endl;
    }

    // 3. Make a convergence graph.
}

template <class T>
void gsLowRankFitting<T>::computeCross(bool pivot)
{
    gsBSplineBasis<T> uBasis = *(static_cast<gsBSplineBasis<T>*>(&(this->m_basis->component(0))));
    gsBSplineBasis<T> vBasis = *(static_cast<gsBSplineBasis<T>*>(&(this->m_basis->component(1))));

    gsMatrix<real_t> coefs(uBasis.size(), vBasis.size());
    coefs.setZero();

    // 0. Convert points to an m x n matrix.
    gsMatrix<T> uPar, vPar;
    index_t uNum = partitionParam(uPar, vPar);
    gsMatrix<real_t> ptsMN = convertToMN(uNum);

    gsMatrixCrossApproximation<T> crossApp(ptsMN);
    T sigma;
    gsVector<T> uVec, vVec;
    //index_t iter = 0;
    while(crossApp.nextIteration(sigma, uVec, vVec, pivot))
    {
	//gsInfo << "iter: " << iter++ << std::endl;
	// 2.0. Fit u with the u-basis.
	// gsInfo << "uPar:\n" << uPar << std::endl;
	// gsInfo << "uPts:\n" << uVec.transpose() << std::endl;
	gsFitting<real_t> uFitting(uPar, uVec.transpose(), uBasis);
	uFitting.compute();
	
	// 2.1. Fit v with the v-basis.
	// gsInfo << "vPar:\n" << vPar << std::endl;
	// gsInfo << "vPts:\n" << vVec.transpose() << std::endl;
	gsFitting<real_t> vFitting(vPar, vVec.transpose(), vBasis);
	vFitting.compute();

	// 2.3. Add the coefficients to the running tensor-product B-spline.
	gsVector<real_t> uCoefs = uFitting.result()->coefs().col(0);
	gsVector<real_t> vCoefs = vFitting.result()->coefs().col(0);
	matrixUtils::addTensorProduct(coefs, sigma, uCoefs, vCoefs);

	delete this->m_result;
	this->m_result = this->m_basis->makeGeometry(give(convertBack(coefs).transpose())).release();
	this->computeErrors();
	gsInfo << this->maxPointError() << std::endl;
    }

    // 3. Make a convergence graph.
}

template <class T>
void gsLowRankFitting<T>::computeRes()
{
    gsBSplineBasis<T> uBasis = *(static_cast<gsBSplineBasis<T>*>(&(this->m_basis->component(0))));
    gsBSplineBasis<T> vBasis = *(static_cast<gsBSplineBasis<T>*>(&(this->m_basis->component(1))));

    gsMatrix<real_t> coefs(uBasis.size(), vBasis.size());
    coefs.setZero();

    // 0. Convert points to an m x n matrix.
    gsMatrix<T> uPar, vPar;
    index_t uNum = partitionParam(uPar, vPar);
    gsMatrix<real_t> ptsMN = convertToMN(uNum);

    gsMatrixCrossApproximation<T> crossApp(ptsMN);
    T sigma;
    gsVector<T> uVec, vVec;
    //index_t iter = 0;
    for (index_t rank=0; rank<ptsMN.rows(); rank++)
    {
	crossApp.nextIteration(sigma, uVec, vVec);

	//gsInfo << "iter: " << iter++ << std::endl;
	// 2.0. Fit u with the u-basis.
	// gsInfo << "uPar:\n" << uPar << std::endl;
	// gsInfo << "uPts:\n" << uVec.transpose() << std::endl;
	gsFitting<real_t> uFitting(uPar, uVec.transpose(), uBasis);
	uFitting.compute();
	
	// 2.1. Fit v with the v-basis.
	// gsInfo << "vPar:\n" << vPar << std::endl;
	// gsInfo << "vPts:\n" << vVec.transpose() << std::endl;
	gsFitting<real_t> vFitting(vPar, vVec.transpose(), vBasis);
	vFitting.compute();

	// 2.3. Add the coefficients to the running tensor-product B-spline.
	gsVector<real_t> uCoefs = uFitting.result()->coefs().col(0);
	gsVector<real_t> vCoefs = vFitting.result()->coefs().col(0);
	matrixUtils::addTensorProduct(coefs, sigma, uCoefs, vCoefs);

	delete this->m_result;
	this->m_result = this->m_basis->makeGeometry(give(convertBack(coefs).transpose())).release();
	this->computeErrors();
	gsInfo << this->maxPointError() << std::endl;

	crossApp.updateMatrix(getErrorsMN(uNum));
    }

    // 3. Make a convergence graph.
}

} // namespace gismo
