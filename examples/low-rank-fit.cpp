#include <gismo.h>
#include <gsModeling/gsLowRankFitting.h>
#include <gsModeling/gsL2Error.h>
#include <gsIO/gsWriteGnuplot.h>

#include <random>

using namespace gismo;

gsMatrix<real_t> testMatrix(index_t example = 0)
{
    if(example == 0)
    {
	gsMatrix<real_t> coefs(3, 3);
	coefs.setZero();
	coefs(0, 0) = 1;
	coefs(0, 1) = 2;
	coefs(0, 2) = 3;
	coefs(1, 0) = 3;
	coefs(1, 1) = 2;
	coefs(1, 2) = 1;
	coefs(2, 0) = 1;

	return coefs;
    }
    else if(example == 1)
    {
	gsMatrix<real_t> coefs(4, 4);
	coefs.setZero();
	coefs(0, 0) = 2;
	coefs(0, 1) = 3;
	coefs(1, 0) = -1;
	coefs(1, 1) = 3;
	coefs(1, 2) = 4;
	coefs(2, 2) = 1;
	coefs(2, 3) = -1;
	coefs(3, 3) = 1;

	return coefs;
    }
    else if(example == 2)
    {
	gsMatrix<real_t> coefs(5,5);
	coefs.setZero();
	coefs(0, 4) = 1;
	coefs(1, 3) = -1;
	coefs(2, 2) = 1;
	coefs(3, 1) = -1;
	coefs(4, 0) = 1;

	coefs(0, 0) = 2;
	coefs(1, 1) = 3;
	coefs(3, 3) = -2;
	coefs(4, 4) = -3;

	coefs(1, 4) = 5;
	coefs(2, 3) = 7;
	coefs(3, 4) = -1;
	coefs(4, 3) = 9;

	return coefs;
    }
    else
    {
	index_t numU = 5;
	index_t numV = 5;
	gsMatrix<real_t> coefs(numU, numV);
	for(index_t i = 0; i < numU; i++)
	{
	    for(index_t j = 0; j < numV; j++)
	    {
		real_t u = i / (numU - 1.0);
		real_t v = j / (numV - 1.0);
		coefs(i, j) = (math::exp(math::sqrt(u * u + v * v))) / 4.0;
	    }
	}
	return coefs;
    }
}

bool checkSvd()
{
    gsMatrix<real_t> coefs = testMatrix();
    gsSvd<real_t> svd(coefs);
    return svd.sanityCheck(coefs);
}

void checkCrossApp(index_t example, bool pivot)
{
    gsMatrix<real_t> coefs = testMatrix(example);
    gsMatrix<real_t> check(coefs.rows(), coefs.cols());
    check.setZero();

    gsInfo << "Target:\n" << coefs << std::endl;

    gsMatrixCrossApproximation<real_t> crossApp(coefs);

    real_t sigma;
    gsVector<real_t> uVec, vVec;
    for(index_t i=0; i<coefs.rows(); i++)
    {
	crossApp.nextIteration(sigma, uVec, vVec, pivot);

	matrixUtils::addTensorProduct(check, sigma, uVec, vVec);
	gsInfo << "Iteration " << i << ":\n" 
	       << "check:\n" << check << std::endl;
    }
	
}

void checkCrossAppMat(index_t example, bool pivot)
{
    gsMatrix<real_t> coefs = testMatrix(example);
    gsMatrix<real_t> check(coefs.rows(), coefs.cols());
    check.setZero();

    gsMatrix<real_t> U(coefs.rows(), coefs.cols());
    gsMatrix<real_t> V(coefs.rows(), coefs.cols());
    gsMatrix<real_t> T(coefs.cols(), coefs.cols());
    T.setZero();

    gsInfo << "Target:\n" << coefs << std::endl;

    gsMatrixCrossApproximation<real_t> crossApp(coefs);

    real_t sigma;
    gsVector<real_t> uVec, vVec;
    for(index_t i=0; i<coefs.rows(); i++)
    {
	crossApp.nextIteration(sigma, uVec, vVec, pivot);
	U.col(i) = uVec;
	V.col(i) = vVec;
	T(i, i) = sigma;
    }

    gsInfo << "U:\n" << U << "\nT:\n" << T << "\nV:\n" << V << std::endl;

    gsInfo << "UTV^T:\n" << U * T * V.transpose() << std::endl;
	
}

template <class T>
void sampleData(const gsVector<T>& uPar, const gsVector<T>& vPar,
		gsMatrix<T>& params, gsMatrix<T>& points, index_t sample)
{
    index_t uNum = uPar.size();
    index_t vNum = vPar.size();
    index_t numSamples = uNum * vNum;

    params.resize(2, numSamples);
    points.resize(1, numSamples);

    // gsInfo << "uPar:\n";
    // for(index_t i=0; i<uPar.size(); i++)
    // 	gsInfo << uPar(i) << " ";
    // gsInfo << std::endl;

    for(index_t i=0; i<uNum; i++)
    {
	for(index_t j=0; j<vNum; j++)
	{
	    index_t glob = j * vNum + i;
	    real_t u = uPar(i);
	    real_t v = vPar(j);

	    params(0, glob) = u;
	    params(1, glob) = v;

	    points(0, glob) = evalSample(u, v, sample);
	    //gsInfo << points(0, glob) << std::endl;
	}
    }
    // gsInfo << "params:\n" << params << std::endl;
    // gsInfo << "points:\n" << points << std::endl;
}

template <class T>
gsVector<T> sampleUniform(index_t size, T tMin, T tMax)
{
    gsVector<T> res(size);
    for(index_t i=0; i<size; i++)
    {
	real_t tLoc = i / (size - 1.0);
	real_t tAct = tMin * (1 - tLoc) + tMax * tLoc;
	res(i) = tAct;
    }
    return res;
}

template <class T>
void sampleData(index_t uNum, index_t vNum, gsMatrix<T>& params, gsMatrix<T>& points,
		index_t sample,
		T uMin = 0, T vMin = 0, T uMax = 1, T vMax = 1)
{
    gsVector<T> uPar = sampleUniform(uNum, uMin, uMax);
    gsVector<T> vPar = sampleUniform(vNum, vMin, vMax);

    sampleData(uPar, vPar, params, points, sample);
}

template <class T>
void sampleData(index_t numSide, gsMatrix<T>& params, gsMatrix<T>& points,
		index_t sample, T tMin = 0, T tMax = 1)
{
    sampleData(numSide, numSide, params, points, sample, tMin, tMin, tMax, tMax);
}

template <class T>
void sampleDataGre(const gsKnotVector<T>& knotsU, const gsKnotVector<T>& knotsV,
		   gsMatrix<T>& params, gsMatrix<T>& points, index_t sample)
{
    gsMatrix<T> greU, greV;
    knotsU.greville_into(greU);
    knotsV.greville_into(greV);

    gsVector<T> uPar = greU.row(0);
    gsVector<T> vPar = greV.row(0);

    sampleData(uPar, vPar, params, points, sample);
}

template <class T>
void sampleDataGre(index_t numSide, gsMatrix<T>& params, gsMatrix<T>& points,
		   index_t sample, T minT = 0, T maxT = 1, index_t deg = 3)
{
    gsKnotVector<T> kv(minT, maxT, numSide - deg - 1, deg + 1);
    sampleDataGre(kv, kv, params, points, sample);
}

template <class T>
void sampleParamsAndWeights(const gsBSplineBasis<T>& basis,
			    gsVector<T>& params,
			    gsVector<T>& weights,
			    T quA = 1.0,
			    index_t quB = 1,
			    bool verbose = false)
{
    gsOptionList legendreOpts;
    legendreOpts.addInt   ("quRule","Quadrature rule used (1) Gauss-Legendre; (2) Gauss-Lobatto; (3) Patch-Rule",gsQuadrature::GaussLegendre);
    legendreOpts.addReal("quA", "Number of quadrature points: quA*deg + quB", quA);
    legendreOpts.addInt ("quB", "Number of quadrature points: quA*deg + quB", quB);
    legendreOpts.addSwitch("overInt","Apply over-integration or not?",false);
    gsQuadRule<real_t>::uPtr legendre = gsQuadrature::getPtr(basis, legendreOpts);

    params.clear();
    weights.clear();

    gsMatrix<T> locParams, globParams(1, 0);
    gsVector<T> locWeights;
    index_t start;

    for (auto domIt = basis.makeDomainIterator(); domIt->good(); domIt->next() )
    {
	if(verbose)
	{
	    gsInfo<<"---------------------------------------------------------------------------\n";
	    gsInfo  <<"Element with corners (lower) "
		    <<domIt->lowerCorner().transpose()<<" and (higher) "
		    <<domIt->upperCorner().transpose()<<" :\n";
	}

	// Gauss-Legendre rule (w/o over-integration)
	legendre->mapTo(domIt->lowerCorner(), domIt->upperCorner(),
			locParams, locWeights);

	if (verbose)
	{
	    gsInfo  << "* \t Gauss-Legendre\n"
		    << "- points:\n"  << locParams              <<"\n"
		    << "- weights:\n" << locWeights.transpose() <<"\n";
	}

	// Append locParams to globParams.
	start = globParams.cols();
	globParams.conservativeResize(Eigen::NoChange, start + locParams.cols());
	globParams.block(0, start, globParams.rows(), locParams.cols()) = locParams;

	// Append locWeights to weights.
	weights.conservativeResize(weights.size() + locWeights.size());
	for(index_t j=0; j<locWeights.size(); j++)
	    weights(start + j) = locWeights(j);
    }

    params = globParams.row(0);
    //gsInfo << "params:\n" << params << std::endl;
}

template <class T>
void sampleDataGauss(const gsTensorBSplineBasis<2, T>& basis,
		     gsMatrix<T>& params,
		     gsMatrix<T>& points,
		     gsVector<T>& uWeights,
		     gsVector<T>& vWeights,
		     index_t sample,
		     T       quA = 1.0,
		     index_t quB = 1)
{
    bool verbose = false;

    const gsBSplineBasis<T> uBasis = basis.component(0);
    const gsBSplineBasis<T> vBasis = basis.component(1);

    gsVector<T> uParams, vParams;

    sampleParamsAndWeights(uBasis, uParams, uWeights, quA, quB, verbose);
    sampleParamsAndWeights(vBasis, vParams, vWeights, quA, quB, verbose);

    GISMO_ASSERT(uParams.size() == uWeights.size(), "Different number of params and weights in u-direction.");
    GISMO_ASSERT(vParams.size() == vWeights.size(), "Different number of params and weights in v-direction.");

    // FIXME:
    GISMO_ASSERT(uParams.size() == vParams.size(), "Problem not symmetric. This is possible in theory but not in the current implementation.");

    // for(index_t i=0; i<uWeights.size(); i++)
    // {
    // 	uWeights(i) = math::sqrt(uWeights(i));
    // 	vWeights(i) = math::sqrt(vWeights(i));
    // }

    sampleData(uParams, vParams, params, points, sample);
}

real_t stdFit(const gsMatrix<real_t>& params,
	      const gsMatrix<real_t>& points,
	      index_t numKnots,
	      index_t deg,
	      index_t sample,
	      real_t minU = 0.0,
	      real_t maxU = 1.0)
{
    gsKnotVector<real_t> knots(minU, maxU, numKnots, deg+1);
    gsTensorBSplineBasis<2, real_t> basis(knots, knots);
    
    gsFitting<real_t> fitting(params, points, basis);
    fitting.compute();
    fitting.computeErrors();
    gsInfo << "Max err of standard fitting: " << fitting.maxPointError() << std::endl;
    //gsInfo << "L2  err of standard fitting: " << fitting.L2Error() << std::endl;
    real_t L2Err = L2Error(*static_cast<gsTensorBSpline<2, real_t>*>(fitting.result()), sample);
    gsInfo << "L2 error of standard fitting: "
	   << L2Err
	   << std::endl;
    //gsWriteParaview(*fitting.result(), "fitting", 10000, false, true);
    // gsFileData<real_t> fd;
    // fd << *fitting.result();
    // fd.dump("fitting");
    return L2Err;
}

void lowSVDFit(const gsMatrix<real_t>& params,
	       const gsMatrix<real_t>& points,
	       index_t numKnots,
	       index_t deg,
	       index_t sample,
	       index_t maxIter,
	       const std::string& filename,
	       real_t minU = 0.0,
	       real_t maxU = 1.0)
{
    gsKnotVector<> knots(minU, maxU, numKnots, deg+1);
    gsTensorBSplineBasis<2> basis(knots, knots);

    gsInfo << "SVD fitting:\n";
    gsLowRankFitting<real_t> fitting(params, points, basis);
    fitting.computeSVD(maxIter, sample, filename);

    //gsWriteParaview(*fitting.result(), "low-rank", 10000, false, true);

    // gsFileData<real_t> fd;
    // fd << *fitting.result();
    // fd.dump("low-rank");
}

void lowCrossAppFit(const gsMatrix<real_t>& params,
		    const gsMatrix<real_t>& points,
		    index_t numKnots,
		    index_t deg,
		    index_t sample,
		    index_t maxIter,
		    const std::string& filename,
		    bool pivot,
		    real_t minU = 0.0,
		    real_t maxU = 1.0)
{
    gsKnotVector<> knots(minU, maxU, numKnots, deg+1);
    gsTensorBSplineBasis<2> basis(knots, knots);

    gsLowRankFitting<real_t> fitting(params, points, basis);
    gsInfo << "CrossApp fitting";
    if(pivot)
	gsInfo << " with pivoting";
    gsInfo << ":\n";
    fitting.computeCross(pivot, maxIter, sample);
    if(pivot)
	fitting.exportL2Err(filename + "piv_L2.dat");
    else
	fitting.exportL2Err(filename + "full_L2.dat");

    //gsWriteParaview(*fitting.result(), "low-rank", 10000, false, true);

    // gsFileData<real_t> fd;
    // fd << *fitting.result();
    // fd.dump("low-rank");
}

void lowCrossResFit(const gsMatrix<real_t>& params,
		    const gsMatrix<real_t>& points,
		    index_t numKnots,
		    index_t deg,
		    real_t minU = 0.0,
		    real_t maxU = 1.0)
{
    gsKnotVector<> knots(minU, maxU, numKnots, deg+1);
    gsTensorBSplineBasis<2> basis(knots, knots);

    gsLowRankFitting<real_t> fitting(params, points, basis);
    gsInfo << "CrossApp residual fitting:\n";
    fitting.computeRes();

    //gsWriteParaview(*fitting.result(), "low-rank", 10000, false, true);

    // gsFileData<real_t> fd;
    // fd << *fitting.result();
    // fd.dump("low-rank");
}

void param()
{
    gsFileData<> fd("example-2-rank2.xml");
    gsTensorBSpline<2> bspline;
    fd.getId<gsTensorBSpline<2>>(0, bspline);
    gsVector<real_t> shift(2);
    shift << -0.5, -0.5;
    bspline.translate(shift);
    //bspline.rotate(5 * EIGEN_PI / 14);
    gsWriteParaview(bspline, "bspline", 1000, false, true);

    gsBSpline<> bBott, bLeft, bRght;
    bspline.slice(1, 0.0, bBott);
    bspline.slice(0, 0.0, bLeft);
    bspline.slice(0, 1.0, bRght);

    gsMatrix<> cBott = bBott.coefs();
    gsMatrix<> cLeft = bLeft.coefs();
    gsMatrix<> cRght_0 = bRght.coefs();

    gsInfo << "slice:\n" << cRght_0 << std::endl;
    // Get-around, since slice gives wrong results for par = 1.0,
    // cf. https://github.com/gismo/gismo/issues/504
    gsMatrix<> cRght(5, 2), cTopp(5, 2);
    gsMatrix<> coefs = bspline.coefs();
    for(index_t i=0; i<5; i++)
    {
	for(index_t j=0; j<2; j++)
	{
	    cRght(i, j) = coefs(5 * i + 4, j);
	    cTopp(i, j) = coefs(20 + i, j);
	}
    }
    gsInfo << "manual:\n" << cRght << std::endl;
    //gsInfo << bspline.coefs().rows() << " x " << bspline.coefs().cols() << std::endl;

    // Compatibility check:
    // gsInfo << cBott.row(0) << std::endl << cLeft.row(0) << std::endl;
    // gsInfo << cTopp.row(4) << std::endl << cRght.row(4) << std::endl;

    // Figuring out the axial shift (do it on the original and change the sign!).
    //gsInfo << 0.25 * (cBott.row(0) + cBott.row(4) + cTopp.row(0) + cTopp.row(4)) << std::endl;

    gsLowRankFitting<real_t> fitting;
    fitting.CR2I_old(cBott, cLeft, cRght, cTopp);
    fitting.CR2I_new(cBott, cLeft, cRght, cTopp);
}

void development()
{
    //checkSvd();
    gsMatrix<real_t> params, points;
    real_t minT = -1.0; // -1 leads to a confusion index_t / real_t.
    index_t sample = 4;
    //real_t minT = 0;
    //sampleData(10, params, points, 5, minT);
    sampleDataGre(10, params, points, sample, minT, 1.0, 2);
    //sampleDataGre(50, params, points, 6, minT, 1.0, 2);
    // Experience: for examples 0 and 1 (rank 1 and 2, respectively),
    // we obtain the same precision as the standard fit after rank
    // iterations. Cool! Can we prove this to be true in general?

    index_t numKnots = 0;
    index_t deg = 2;
    index_t maxIter = 5;
    std::string filename = "old";
    stdFit(        params, points, numKnots, deg, sample, minT);
    lowSVDFit(     params, points, numKnots, deg, sample, maxIter, filename, minT);
    lowCrossAppFit(params, points, numKnots, deg, sample, maxIter, filename, false, minT);
    lowCrossAppFit(params, points, numKnots, deg, sample, maxIter, filename, true,  minT);
    //lowCrossResFit(params, points, numKnots, deg);
    //checkCrossApp(false);
    //checkCrossApp(3, true);
    //checkCrossAppMat(true);

    //param();
}

void example_1()
{
    gsMatrix<real_t> params, points;
    real_t minT = -1.0;
    index_t sample = 4;
    sampleData(500, params, points, sample, minT);

    std::vector<index_t> numKnots(2);
    numKnots[0] = 46;
    numKnots[1] = 96;

    // for(auto it=dataSizes.begin(); it!=dataSizes.end(); ++it)
    // {
    // 	// TODO: Hack here.
    // }
}

void example_2()
{
    index_t sample = 6;
    std::vector<index_t> dataSizes(2);
    dataSizes[0] = 50;
    dataSizes[1] = 100;
    //dataSizes[2] = 200;
    //dataSizes[3] = 400;

    for(auto it=dataSizes.begin(); it!=dataSizes.end(); ++it)
    {
	gsMatrix<real_t> params, points;
	real_t minT = -1.0; // -1 leads to a confusion index_t / real_t.
	sampleDataGre(*it, params, points, sample, minT, 1.0, 2);

	index_t deg = 2;
	index_t numKnots = *it - deg - 1;
	index_t maxIter = 25;

	std::string filename = std::to_string(*it);

	//stdFit(        params, points, numKnots, deg, minT);
	// lowSVDFit(     params, points, numKnots, deg, maxIter, filename, minT);
	//lowCrossAppFit(params, points, numKnots, deg, maxIter, filename, false, minT);
	lowCrossAppFit(params, points, numKnots, deg, sample, maxIter, filename, true,  minT);
    }
}

void example_3()
{
    // Integrate 1/4 exp(sqrt(x^2+y^2) over the unit square.
    // The correct answer (from Wolfram cloud) is 0.55886658581726677503.
    index_t numDOF = 100;

    // 1. Take a tensor-product basis for fitting.
    index_t deg = 3;
    index_t numKnots = numDOF - deg -1;
    real_t tMin = -1.0;
    real_t tMax =  1.0;
    gsKnotVector<real_t> kv(tMin, tMax, numKnots, deg + 1);
    gsTensorBSplineBasis<2, real_t> fittingBasis(kv, kv);

    // 2. Sample points and weights for a high-order quadrature rule.
    index_t sample = 6;
    gsVector<real_t> uWeights, vWeights;
    gsMatrix<real_t> params, points;
    sampleDataGauss<real_t>(fittingBasis, params, points, uWeights, vWeights, sample, 2.0, 3);
    gsInfo << "Sampled " << params.cols() << " points, fitting with "
	   << fittingBasis.size() << " DOF." << std::endl;

    // 3. Approximate in the least-squares sense.
    index_t maxIter = 20;
    std::string filename = "example-3-case-6-weighted";

    gsLowRankFitting<real_t> fitting(params, points, uWeights, vWeights, fittingBasis);
    gsInfo << "CrossApp fitting with weights and pivoting:\n";
    //gsInfo << "Params (" << params.rows() << "x" << params.cols() << "):\n" << params << std::endl;
    //gsInfo << "Points (" << points.rows() << "x" << points.cols() << "):\n" << points << std::endl;
    fitting.computeCross(true, maxIter, sample);
    fitting.exportL2Err(filename + "piv_L2.dat");

    // 4. Compare with the full fitting without weights.
    stdFit(params, points, numKnots, deg, sample, tMin);

    // 5. Compare with Irina & Clemens using the same spline space.
    sampleDataGre(numDOF, params, points, sample, tMin, tMax, deg);
    gsInfo << "Sampled " << params.cols() << " points, fitting with "
	   << fittingBasis.size() << " DOF." << std::endl;
    filename = "example-3-case-6-inter100";
    lowCrossAppFit(params, points, numKnots, deg, sample, maxIter, filename, true, tMin);

    // 6. And finally full fitting in these points.
    stdFit(params, points, numKnots, deg, sample, tMin);
}

void example_4()
{
    index_t sample = 4; // TODO: Choose!
    index_t deg = 3;
    index_t maxIter = 10;
    real_t tMin = -1;
    real_t tMax = 1;
    index_t nExperiments = 6;

    gsMatrix<real_t> params, points;

    std::vector<index_t> numDOF(nExperiments);
    numDOF[0] = 16;
    numDOF[1] = 32;
    numDOF[2] = 64;
    numDOF[3] = 128;
    numDOF[4] = 256;
    numDOF[5] = 512;
    sampleDataGre(512, params, points, sample, tMin, tMax, deg);

    std::vector<real_t> stdL2Err;
    std::vector<std::vector<real_t>> lowL2Err;
    std::vector<real_t> knotSpans;
    for(auto it=numDOF.begin(); it!=numDOF.end(); ++it)
    {
	index_t numKnots = *it - deg - 1;
	stdL2Err.push_back(stdFit(params, points, numKnots, deg, sample, tMin));

	gsKnotVector<real_t> knots(tMin, tMax, numKnots, deg+1);
	// Remember h.
	knotSpans.push_back(knots.maxIntervalLength());
	gsTensorBSplineBasis<2, real_t> basis(knots, knots);
	gsLowRankFitting<real_t> lowRankFitting(params, points, basis);
	lowRankFitting.computeCross(true, maxIter, sample);
	lowL2Err.push_back(lowRankFitting.getL2Err());
    }

    // Compute total number of DOF.
    std::vector<real_t> totDOF(numDOF.size());
    for(size_t i=0; i<numDOF.size(); i++)
	totDOF[i] = numDOF[i] * numDOF[i];

    gsWriteGnuplot(totDOF, stdL2Err, "example-4-std.dat");

    for(index_t j=0; j<maxIter; j++)
    {
	std::vector<real_t> curr;
	for(size_t i=0; i<numDOF.size(); i++)
	    if(size_t(j) < lowL2Err[i].size())
		curr.push_back(lowL2Err[i][j]);

	if(curr.size() > 0)
	    gsWriteGnuplot(totDOF, curr, "example-4-rank-" + std::to_string(j+1) + ".dat");
    }

    // Print the LaTeX table with errors.
    gsInfo << "rank \\textbackslash $h$";
    for(auto it=knotSpans.begin(); it!=knotSpans.end(); ++it)
	gsInfo << " & " << *it;
    gsInfo << " \\\\" << std::endl;

    for(index_t j=0; j<maxIter; j++)
    {
	gsInfo << j+1 << " & ";
	for(index_t i=0; i<nExperiments; i++)
	{
	    if(size_t(j) < lowL2Err[i].size())
		gsInfo << lowL2Err[i][j];
	    else
		gsInfo << " - ";
	    if(i == nExperiments - 1)
		gsInfo << "\\\\" << std::endl; // We need quadruple backslash, because a double backslash prints as a single backslash.
	    else
		gsInfo << " & ";
	}
    }
}

void example_5()
{
    index_t sample = 4; // TODO: Choose!
    index_t deg = 3;
    real_t tMin = -1;
    real_t tMax = 1;

    gsMatrix<real_t> params, points;

    index_t numDOF = 32;

    std::vector<real_t> stdL2Err;
    index_t numKnots = numDOF - deg - 1;
    // TODO: Sampling 32 points and fitting them with 32 DOF does not converge (!).
    sampleDataGre(512, params, points, sample, tMin, tMax, deg);

    gsInfo << "params: " << params.rows() << " x " << params.cols() << std::endl;

    gsKnotVector<real_t> knots(tMin, tMax, numKnots, deg+1);
    gsTensorBSplineBasis<2, real_t> basis(knots, knots);

    gsInfo << "basis:\n" << basis << std::endl;
    gsLowRankFitting<real_t> lowRankFitting(params, points, basis);
    lowRankFitting.computeCross(true, 50, sample);

    stdFit(params, points, numKnots, deg, sample, tMin);
}

void example_6()
{
    index_t sample  = 8; // TODO: It would be good to have a function not from Irina & Clemens.
    index_t deg = 3;
    real_t tMin = 0;
    real_t tMax = 2;
    index_t nExperiments = 7;
    index_t nRanks = 5;
    real_t delta = 3;

    gsMatrix<real_t> params, points;

    std::vector<index_t> numDOF(nExperiments);
    if(nExperiments > 0) numDOF[0] = 32;
    if(nExperiments > 1) numDOF[1] = 64;
    if(nExperiments > 2) numDOF[2] = 128;
    if(nExperiments > 3) numDOF[3] = 256;
    if(nExperiments > 4) numDOF[4] = 512;
    if(nExperiments > 5) numDOF[5] = 1024;
    if(nExperiments > 6) numDOF[6] = 2048;

    std::vector<index_t> maxRanks(nRanks);
    if(nRanks > 0) maxRanks[0] = 4;
    if(nRanks > 1) maxRanks[1] = 8;
    if(nRanks > 2) maxRanks[2] = 16;
    if(nRanks > 3) maxRanks[3] = 32;
    if(nRanks > 4) maxRanks[4] = 64;

    std::vector<real_t> timesB;
    gsMatrix<real_t> timesC(numDOF.size(), maxRanks.size());
    gsStopwatch time;
    for(size_t i=0; i<numDOF.size(); i++)
    {
	sampleDataGre(delta * numDOF[i], params, points, sample, tMin, tMax, deg);
	index_t numKnots = numDOF[i] - deg - 1;

	gsKnotVector<real_t> knots(tMin, tMax, numKnots, deg+1);
	gsTensorBSplineBasis<2, real_t> basis(knots, knots);
	gsLowRankFitting<real_t> lowRankFitting(params, points, basis);

	gsInfo << "Method B:\n";
	timesB.push_back(lowRankFitting.methodB());

	for(size_t j=0; j<maxRanks.size(); j++)
	{
	    index_t maxRank = maxRanks[j];
	    gsInfo << "Method C, rank " << maxRank << ":\n";
	    timesC(i, j) = lowRankFitting.methodC(maxRank);
	}
    }

    gsWriteGnuplot(numDOF, timesB, "example-6-method-B-fast.dat");

    for(size_t j=0; j<maxRanks.size(); j++)
    {
	std::vector<real_t> timesCurr(numDOF.size());
	for(size_t i=0; i<numDOF.size(); i++)
	    timesCurr[i] = timesC(i, j);
	gsWriteGnuplot(numDOF, timesCurr, "example-6-method-C-fast-iter-" + std::to_string(maxRanks[j]) + ".dat");
    }
}

int main()
{
    //development();
    //example_1();
    //example_2();
    //example_3();
    //example_4();
    //example_5();
    example_6();
    return 0;    
}
