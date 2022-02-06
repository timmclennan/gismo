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
#include <gsModeling/gsL2Error.h>

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
	: gsFitting<T>(params, points, basis),
	  m_uWeights(matrixUtils::diag(uWeights)),
	  m_vWeights(matrixUtils::diag(vWeights))
    {
    }

    gsLowRankFitting(const gsMatrix<T>& params,
		     const gsMatrix<T>& points,
		     gsTensorBSplineBasis<2, T>& basis)
	: gsFitting<T>(params, points, basis)
    {
	// Note: Forgetting the <T> leads to a confusing error message.
	index_t uNpts = math::sqrt(params.cols());
	m_uWeights = matrixUtils::identity<T>(uNpts);
	m_vWeights = matrixUtils::identity<T>(uNpts);
    }

    void computeCross(bool pivot, index_t maxIter, index_t sample, const std::string& filename);
    
    void computeSVD(index_t maxIter, index_t sample, const std::string& filename);

    void computeRes();

    void CR2I_old(const gsMatrix<T>& bott,
		  const gsMatrix<T>& left,
		  const gsMatrix<T>& rght,
		  const gsMatrix<T>& topp) const;

    void CR2I_new(const gsMatrix<T>& bott,
		  const gsMatrix<T>& left,
		  const gsMatrix<T>& rght,
		  const gsMatrix<T>& topp) const;

    //T L2error() const;

protected:

    void gsWriteGnuplot(const std::vector<T>& xData,
			const std::vector<T>& yData,
			const std::string& filename) const;

    void gsWriteGnuplot(const std::vector<T>& data, const std::string& filename) const;

    index_t partitionParam(gsMatrix<T>& uPar, gsMatrix<T>& vPar) const;

    gsMatrix<T> convertToMN(index_t rows) const;

    gsMatrix<T> convertBack(const gsMatrix<T>& points) const;

    gsMatrix<T> getErrorsMN(size_t rows) const;

protected:

    gsMatrix<T> m_uWeights, m_vWeights;
};

template <class T>
void gsLowRankFitting<T>::gsWriteGnuplot(const std::vector<T>& xData,
					 const std::vector<T>& yData,
					 const std::string& filename) const
{
    GISMO_ASSERT(xData.size() == yData.size(), "differring data sizes when writing to gnuplot");

    std::ofstream fout;
    fout.open(filename, std::ofstream::out);
    fout << "# x y\n";
    for(size_t i=0; i<xData.size(); i++)
    	fout << "  " << xData[i] << "   " << yData[i] << "\n";
    fout.close();	
}

template <class T>
void gsLowRankFitting<T>::gsWriteGnuplot(const std::vector<T>& data, const std::string& filename) const
{
    // Note: before converting this to use the overloaded function
    // think about data types (T vs. index_t).

    std::ofstream fout;
    fout.open(filename, std::ofstream::out);
    fout << "# x y\n";
    for(size_t i=0; i<data.size(); i++)
    	fout << "  " << i+1 << "   " << data[i] << "\n";
    fout.close();
}


template <class T>
index_t gsLowRankFitting<T>::partitionParam(gsMatrix<T>& uPar, gsMatrix<T>& vPar) const
{
    index_t uNum = 0;
    T prevMax = -1 * std::numeric_limits<T>::max();
    for(index_t i=0; this->m_param_values(0, i) > prevMax; i++)
    {
	uNum++;
	prevMax = this->m_param_values(0, i);
    }
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
	result(i%rows, i/rows) = this->m_points(i, 0) - val_i(0, i); // math::abs?
    // Signed distance, works in 1D only.
	    //this->m_pointErrors[i];

    return result;
}

template <class T>
void gsLowRankFitting<T>::computeSVD(index_t maxIter, index_t sample, const std::string& filename)
{
    
    gsBSplineBasis<T> uBasis = *(static_cast<gsBSplineBasis<T>*>(&(this->m_basis->component(0))));
    gsBSplineBasis<T> vBasis = *(static_cast<gsBSplineBasis<T>*>(&(this->m_basis->component(1))));

    gsMatrix<T> coefs(uBasis.size(), vBasis.size());
    coefs.setZero();

    // 0. Convert points to an m x n matrix.
    gsMatrix<T> uPar, vPar;
    index_t uNum = partitionParam(uPar, vPar);
    gsMatrix<T> ptsMN = convertToMN(uNum);

    // 1. Perform an SVD on it.
    gsSvd<T> pointSVD(ptsMN);
    //pointSVD.sanityCheck(ptsMN);
    //gsInfo << "Rank: " << pointSVD.rank() << std::endl;

    // 2. Iterate.
    std::vector<T> LIErr;
    std::vector<T> L2Err;
    std::vector<T> dofs;
    for(index_t r=0; r<pointSVD.rank() && r<maxIter; r++)
    {
	// 2.0. Fit u with the u-basis.
	gsFitting<T> uFitting(uPar, pointSVD.u(r).transpose(), uBasis);
	uFitting.compute();
	// A note to future self: Forgetting about the transpose lead to preconditioners failing.

	// 2.1. Fit v with the v-basis.
	gsFitting<T> vFitting(vPar, pointSVD.v(r).transpose(), vBasis);
	vFitting.compute();

	// 2.3. Add the coefficients to the running tensor-product B-spline.
	gsVector<T> uCoefs = uFitting.result()->coefs().col(0);
	gsVector<T> vCoefs = vFitting.result()->coefs().col(0);
	coefs += pointSVD.s(r) * matrixUtils::tensorProduct(uCoefs, vCoefs);

	delete this->m_result;
	this->m_result = this->m_basis->makeGeometry(give(convertBack(coefs).transpose())).release();
	this->computeErrors();
	T maxErr = this->maxPointError();
	gsInfo << "err SVD: " << maxErr << std::endl;
	LIErr.push_back(maxErr);
	L2Err.push_back(L2Error(*static_cast<gsTensorBSpline<2, T>*>(this->result()), sample));
	dofs.push_back((r+1) * uCoefs.size() * vCoefs.size());
    }

    // 3. Make a convergence graph.
    gsWriteGnuplot(LIErr, filename + "svd_max.dat");
    gsWriteGnuplot(L2Err, filename + "svd_L2.dat");
    //gsWriteGnuplot(dofs, L2Err, "svdL2.dat");
}

template <class T>
void gsLowRankFitting<T>::computeCross(bool pivot,
				       index_t maxIter, index_t sample,
				       const std::string& filename)
{
    gsBSplineBasis<T> uBasis = *(static_cast<gsBSplineBasis<T>*>(&(this->m_basis->component(0))));
    gsBSplineBasis<T> vBasis = *(static_cast<gsBSplineBasis<T>*>(&(this->m_basis->component(1))));

    gsMatrix<T> coefs(uBasis.size(), vBasis.size());
    coefs.setZero();

    gsMatrix<T> uPar, vPar;
    index_t uNum = partitionParam(uPar, vPar);
    gsMatrix<T> ptsMN = convertToMN(uNum);

    gsSparseMatrix<T> Xs, Ys;
    uBasis.collocationMatrix(uPar, Xs);
    vBasis.collocationMatrix(vPar, Ys);
    gsMatrix<T> X(Xs.transpose());
    gsMatrix<T> Y(Ys.transpose());

    gsMatrix<T> uLeast = (X * m_uWeights * X.transpose()).inverse() * X * m_uWeights;
    gsMatrix<T> vLeast = (Y * m_vWeights * Y.transpose()).inverse() * Y * m_vWeights;

    gsMatrix<T> U(ptsMN.rows(), ptsMN.cols());
    gsMatrix<T> V(ptsMN.rows(), ptsMN.cols());
    gsMatrix<T> TT(ptsMN.cols(), ptsMN.cols());
    U.setZero();
    TT.setZero();
    V.setZero();
    
    gsMatrixCrossApproximation<T> crossApp(ptsMN);
    T sigma;
    gsVector<T> uVec, vVec;
    std::vector<T> LIErr;
    std::vector<T> L2Err;
    for(index_t i=0; i<maxIter && crossApp.nextIteration(sigma, uVec, vVec, pivot); i++)
    {
	// U.col(i) = uVec;
	// V.col(i) = vVec;
	// TT(i ,i) = sigma;

	gsMatrix<T> uMat(uVec.size(), 1);
	gsMatrix<T> vMat(vVec.size(), 1);
	uMat.col(0) = uVec;
	vMat.col(0) = vVec;

	coefs = coefs + sigma * (uLeast * uMat * (vLeast * vMat).transpose());

	//}
	//coefs = uLeast * U * TT * V.transpose() * vLeast.transpose();
	delete this->m_result;
	this->m_result = this->m_basis->makeGeometry(give(convertBack(coefs).transpose())).release();
	this->computeErrors();
	T maxErr = this->maxPointError();
	T l2Err = L2Error(*static_cast<gsTensorBSpline<2, T>*>(this->result()), sample);
	gsInfo << "max err piv: " << maxErr << ", L2 err piv: " << l2Err << std::endl;
	LIErr.push_back(maxErr);
	L2Err.push_back(l2Err);
	//L2Err.push_back(this->L2Error());
    }

    if(pivot)
    {
	gsWriteGnuplot(LIErr, filename + "piv_max.dat");
	gsWriteGnuplot(L2Err, filename + "piv_L2.dat");
    }
    else
    {
	gsWriteGnuplot(LIErr, filename + "full_max.dat");
	gsWriteGnuplot(L2Err, filename + "full_L2.dat");
    }	
    //gsWriteParaview(*this->m_result, "result");
}

template <class T>
void gsLowRankFitting<T>::computeRes()
{
    gsBSplineBasis<T> uBasis = *(static_cast<gsBSplineBasis<T>*>(&(this->m_basis->component(0))));
    gsBSplineBasis<T> vBasis = *(static_cast<gsBSplineBasis<T>*>(&(this->m_basis->component(1))));

    gsMatrix<T> coefs(uBasis.size(), vBasis.size());
    coefs.setZero();

    // 0. Convert points to an m x n matrix.
    gsMatrix<T> uPar, vPar;
    index_t uNum = partitionParam(uPar, vPar);
    gsMatrix<T> ptsMN = convertToMN(uNum);

    gsMatrixCrossApproximation<T> crossApp(ptsMN);
    T sigma = 0;
    gsVector<T> uVec, vVec;
    //index_t iter = 0;
    for (index_t rank=0; rank<ptsMN.rows(); rank++)
    {
	crossApp.nextIteration(sigma, uVec, vVec, false);

	//gsInfo << "iter: " << iter++ << std::endl;
	// 2.0. Fit u with the u-basis.
	// gsInfo << "uPar:\n" << uPar << std::endl;
	// gsInfo << "uPts:\n" << uVec.transpose() << std::endl;
	gsFitting<T> uFitting(uPar, uVec.transpose(), uBasis);
	uFitting.compute();
	
	// 2.1. Fit v with the v-basis.
	// gsInfo << "vPar:\n" << vPar << std::endl;
	// gsInfo << "vPts:\n" << vVec.transpose() << std::endl;
	gsFitting<T> vFitting(vPar, vVec.transpose(), vBasis);
	vFitting.compute();

	// 2.3. Add the coefficients to the running tensor-product B-spline.
	gsVector<T> uCoefs = uFitting.result()->coefs().col(0);
	gsVector<T> vCoefs = vFitting.result()->coefs().col(0);
	matrixUtils::addTensorProduct(coefs, sigma, uCoefs, vCoefs);

	delete this->m_result;
	this->m_result = this->m_basis->makeGeometry(give(convertBack(coefs).transpose())).release();
	this->computeErrors();
	//gsInfo << this->maxPointError() << std::endl;

	crossApp.updateMatrix(getErrorsMN(uNum));
    }

    // 3. Make a convergence graph.
}

template <class T>
void gsLowRankFitting<T>::CR2I_old(const gsMatrix<T>& bott,
				   const gsMatrix<T>& left,
				   const gsMatrix<T>& rght,
				   const gsMatrix<T>& topp) const
{
    gsMatrix<T> res(25, 2);

    for(index_t k=0; k<2; k++)
    {
	gsMatrix<T> c(5, 5);

	// Prepare the boundary.
	for(index_t i=0; i<c.rows(); i++)
	{
	    c(i, 0) = left(i, k);
	    c(i, 4) = rght(i, k);
	}
	for(index_t j=0; j<c.cols(); j++)
	{
	    c(0, j) = bott(j, k);
	    c(4, j) = topp(j, k);
	}

	T delta = matrixUtils::det2x2(c(0, 0), c(0, 4),
				      c(4, 0), c(4, 4));
	if(delta == 0)
	    GISMO_ERROR("Infeasible configuration.");

	for(index_t j=1; j<c.cols() - 1; j++)
	{
	    T lambda = (1.0/delta) * matrixUtils::det2x2(c(0, j), c(0, 4),
							 c(4, j), c(4, 4));

	    T rho    = (1.0/delta) * matrixUtils::det2x2(c(0, 0), c(0, j),
							 c(4, 0), c(4, j));
	    
	    for(index_t i=1; i<c.rows() - 1; i++)
		c(i, j) = lambda * c(i, 0) + rho * c(i, 4);
	}

	// Transcribe from c to res(k).
	for(index_t i=0; i<c.rows(); i++)
	    for(index_t j=0; j<c.cols(); j++)
		res(5 * i + j, k) = c(i, j);
    }

    gsKnotVector<T> KV(0.0, 1.0, 0, 5);
    gsTensorBSpline<2, T> param(KV, KV, res);
    gsWriteParaview(param, "param", 1000, false, true);

    // gsBSpline<T> fBott(0.0, 1.0, 0, 4, bott);
    // gsBSpline<T> fLeft(0.0, 1.0, 0, 4, left);
    // gsBSpline<T> fRght(0.0, 1.0, 0, 4, rght);
    // gsBSpline<T> fTopp(0.0, 1.0, 0, 4, topp);
    // gsWriteParaview(fBott, "bott");
    // gsWriteParaview(fLeft, "left");
    // gsWriteParaview(fRght, "rght");
    // gsWriteParaview(fTopp, "topp");
}

// TODO next time:
// - CR2I_old disagrees with the old example: is it CR2I or AR5I there?
// - Implement CR2I_new.
// - Compare the two.
template <class T>
void gsLowRankFitting<T>::CR2I_new(const gsMatrix<T>& bott,
				   const gsMatrix<T>& left,
				   const gsMatrix<T>& rght,
				   const gsMatrix<T>& topp) const
{
    gsKnotVector<T> KV(0.0, 1.0, 0, 5);
    gsBSplineBasis<T> uBasis(KV);
    gsBSplineBasis<T> vBasis(KV);

    gsBSpline<T> bSpline(uBasis, bott), tSpline(uBasis, topp);
    gsBSpline<T> lSpline(vBasis, left), rSpline(vBasis, rght);

    gsMatrix<T> uPar(1, 5), vPar(1, 5);
    uPar << 0, 0.2, 0.4, 0.6, 0.8;
    vPar << 0, 0.2, 0.4, 0.6, 0.8;

    gsMatrix<T> coefs(uBasis.size(), vBasis.size());
    coefs.setZero();

    gsSparseMatrix<T> Xs, Ys;
    uBasis.collocationMatrix(uPar, Xs);
    vBasis.collocationMatrix(vPar, Ys);
    gsMatrix<T> X(Xs.transpose());
    gsMatrix<T> Y(Ys.transpose());

    gsMatrix<T> uLeast = (X * X.transpose()).inverse() * X;
    gsMatrix<T> vLeast = (Y * Y.transpose()).inverse() * Y;

    gsMatrix<T> res(25, 2);

    // TODO: The following uses CPs but we need to assemble values in uPar and vPar.
    for(index_t k=0; k<2; k++)
    {
	gsMatrix<T> U(5, 2);
	gsMatrix<T> TT(2, 2);
	gsMatrix<T> V(5, 2);
	TT.setZero();

	U.col(0) = bott.col(k);
	U.col(1) = topp.col(k) - bott.col(k);

	TT(0, 0) = 1.0 / bott(0, k);
	TT(1, 1) = 1.0 / (topp(4, k) - TT(0, 0));

	V.col(0) = left.col(k);
	V.col(1) = rght.col(k);

	coefs = uLeast * U * TT * V.transpose() * vLeast.transpose();

	for(index_t i=0; i<5; i++)
	    for(index_t j=0; j<5; j++)
		res(5 * i + j, k) = coefs(i, j);
    }

    gsTensorBSpline<2, T> result(KV, KV, res);
    gsWriteParaview(result, "result_new", 1000, false, true);
}

// template <class T>
// T gsLowRankFitting<T>::L2error() const
// {
//     T result = 0;
//     for(auto it=this->m_pointErrors.begin(); it!=this->m_pointErrors.end(); ++it)
// 	result += (*it) * (*it);

//     return math::sqrt(result);
//     //return result;
// }

} // namespace gismo
