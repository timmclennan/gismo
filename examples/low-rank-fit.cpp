#include <gismo.h>
#include <gsModeling/gsLowRankFitting.h>

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
void sampleData(index_t numU, index_t numV, gsMatrix<T>& params, gsMatrix<T>& points, index_t example)
{
    index_t numSamples = numU * numV;
    params.resize(2, numSamples);
    points.resize(1, numSamples);

    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_real_distribution<> dist(0, 1); // distribution in range [-1, 1]

    for(index_t i=0; i<numU; i++)
    {
	for(index_t j=0; j<numV; j++)
	{
	    index_t glob = j * numV + i;
	    real_t u = i / (numU - 1.0);
	    real_t v = j / (numV - 1.0);

	    params(0, glob) = u;
	    params(1, glob) = v;

	    switch(example)
	    {
	    case 0:
		points(0, glob) = math::sin(u  * 2 * EIGEN_PI) * math::sin(v * 2 * EIGEN_PI) * 0.125;
		break;
	    case 1:
		points(0, glob) = math::sin(u  * 2 * EIGEN_PI) * math::sin(v * 2 * EIGEN_PI) * 0.1
		    + math::cos(u  * 2 * EIGEN_PI) * math::cos(v * 2 * EIGEN_PI) * 0.1;
		break;
	    case 2:
		points(0, glob) = math::sin((u + v) * EIGEN_PI) * 0.125;
		break;
	    case 3:
	    {
		T num = dist(rng);
		//gsInfo << num << " ";
		points(0, glob) = num;
		break;
	    }
	    case 4:
	    {
		T arg = 5 * EIGEN_PI * ((u - 0.2) * (u - 0.2) + (v - 0.0) * (v - 0.0));
		points(0, glob) = (math::sin(arg)) / arg;
		break;
	    }
	    case 5:
		points(0, glob) = (math::exp(math::sqrt(u * u + v * v))) / 4;
		break;
	    default:
		gsWarn << "Unknown example" << example << "." << std::endl;
		break;
	    }
	}
    }
}

template <class T>
void sampleData(index_t numSide, gsMatrix<T>& params, gsMatrix<T>& points, index_t example)
{
    sampleData(numSide, numSide, params, points, example);
}

void stdFit(const gsMatrix<real_t>& params,
	    const gsMatrix<real_t>& points,
	    index_t numKnots,
	    index_t deg)
{
    gsKnotVector<> knots(0.0, 1.0, numKnots, deg+1);
    gsTensorBSplineBasis<2> basis(knots, knots);
    
    gsFitting<real_t> fitting(params, points, basis);
    fitting.compute();
    fitting.computeErrors();
    gsInfo << "Err of standard fittting: " << fitting.maxPointError() << std::endl;
    gsWriteParaview(*fitting.result(), "fitting", 10000, false, true);
    // gsFileData<real_t> fd;
    // fd << *fitting.result();
    // fd.dump("fitting");
}

void lowSVDFit(const gsMatrix<real_t>& params,
	       const gsMatrix<real_t>& points,
	       index_t numKnots,
	       index_t deg)
{
    gsKnotVector<> knots(0.0, 1.0, numKnots, deg+1);
    gsTensorBSplineBasis<2> basis(knots, knots);

    gsInfo << "SVD fitting:\n";
    gsLowRankFitting<real_t> fitting(params, points, basis);
    fitting.computeSVD();

    //gsWriteParaview(*fitting.result(), "low-rank", 10000, false, true);

    // gsFileData<real_t> fd;
    // fd << *fitting.result();
    // fd.dump("low-rank");
}

void lowCrossAppFit(const gsMatrix<real_t>& params,
		    const gsMatrix<real_t>& points,
		    index_t numKnots,
		    index_t deg)
{
    gsKnotVector<> knots(0.0, 1.0, numKnots, deg+1);
    gsTensorBSplineBasis<2> basis(knots, knots);

    gsLowRankFitting<real_t> fitting(params, points, basis);
    gsInfo << "CrossApp fitting:\n";
    fitting.computeCross(false);

    //gsWriteParaview(*fitting.result(), "low-rank", 10000, false, true);

    // gsFileData<real_t> fd;
    // fd << *fitting.result();
    // fd.dump("low-rank");
}

void lowCrossPivFit(const gsMatrix<real_t>& params,
		    const gsMatrix<real_t>& points,
		    index_t numKnots,
		    index_t deg)
{
    gsKnotVector<> knots(0.0, 1.0, numKnots, deg+1);
    gsTensorBSplineBasis<2> basis(knots, knots);

    gsLowRankFitting<real_t> fitting(params, points, basis);
    gsInfo << "CrossApp pivoting fitting:\n";
    fitting.computeCross_2(true);

    //gsWriteParaview(*fitting.result(), "low-rank", 10000, false, true);

    // gsFileData<real_t> fd;
    // fd << *fitting.result();
    // fd.dump("low-rank");
}


void lowCrossResFit(const gsMatrix<real_t>& params,
		    const gsMatrix<real_t>& points,
		    index_t numKnots,
		    index_t deg)
{
    gsKnotVector<> knots(0.0, 1.0, numKnots, deg+1);
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

    gsBSpline<> bBott, bLeft;
    bspline.slice(1, 0.0, bBott);
    bspline.slice(0, 0.0, bLeft);

    gsMatrix<> cBott = bBott.coefs();
    gsMatrix<> cLeft = bLeft.coefs();

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


int main()
{
    //checkSvd();
    gsMatrix<real_t> params, points;
    sampleData(100, params, points, 5);
    // Experience: for examples 0 and 1 (rank 1 and 2, respectively),
    // we obtain the same precision as the standard fit after rank
    // iterations. Cool! Can we prove this to be true in general?

    index_t numKnots = 19;
    index_t deg = 3;
    //stdFit(        params, points, numKnots, deg);
    //lowSVDFit(     params, points, numKnots, deg);
    //lowCrossAppFit(params, points, numKnots, deg);
    //lowCrossPivFit(params, points, numKnots, deg);
    //lowCrossResFit(params, points, numKnots, deg);
    //checkCrossApp(false);
    //checkCrossApp(3, true);
    //checkCrossAppMat(true);

    param();
    return 0;    
}
