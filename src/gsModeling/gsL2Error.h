/** @file gsL2Error.h

    @brief This is a helper file with a few functions that I should
    integrate elsewhere.

    This file is part of the G+Smo library.
    
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.

    Author(s): D. Mokris
*/

#pragma once

#include <gsAssembler/gsQuadRule.h>
#include <gsAssembler/gsQuadrature.h>
#include <gsIO/gsOptionList.h>

namespace gismo
{

    template <class T>
    T evalSample(T u, T v, index_t example)
    {
	switch(example)
	{
	case 0:
	    return math::sin(u  * 2 * EIGEN_PI) * math::sin(v * 2 * EIGEN_PI) * 0.125;
	case 1:
	    return math::sin(u  * 2 * EIGEN_PI) * math::sin(v * 2 * EIGEN_PI) * 0.1
		+ math::cos(u  * 2 * EIGEN_PI) * math::cos(v * 2 * EIGEN_PI) * 0.1;
	case 2:
	    return math::sin((u + v) * EIGEN_PI) * 0.125;
	case 4:
	{
	    T arg = 5 * EIGEN_PI * ((u - 0.2) * (u - 0.2) + (v - 0.0) * (v - 0.0));
	    return math::sin(arg) / arg;
	}
	case 5:
	    return 0.25 * math::exp(math::sqrt(u * u + v * v));
	case 6:
	    return (2.0 / 3) * (math::exp(-1 * math::sqrt((10 * u - 3) * (10 * u - 3)
							  +
							  (10 * v - 3) * (10 * v - 3)))
				+
				math::exp(-1 * math::sqrt((10 * u + 3) * (10 * u + 3)
							  +
							  (10 * v + 3) * (10 * v + 3)))
				);
	case 7:
	    return T(0);
	case 8:
	{
	    if(v - u > 0.5)
		return 1;
	    else if(v - u >= 0) // automatically also <= 0.5
		return 2 * (v - u);
	    else
	    {
		T arg = (u - 1.5) * (u - 1.5) + (v - 0.5) * (v - 0.5);
		if(arg <= 1.0 / 16)
		    return (math::cos(4 * EIGEN_PI * math::sqrt(arg)) + 1) / 2;
		else
		    return 0;
	    }
	}
	case 9:
	{
	    return math::cos(10 * u * (1 + v * v)) / (1 + 10 * (u + 2 * v) * (u + 2 * v));
	}
	default:
	    gsWarn << "Unknown example " << example << "." << std::endl;
	    return 0;
	}
    }

    template <class T>
    T L2Error(const gsTensorBSpline<2, T>& spline,
	      index_t sample,
	      T quA = 2.0,
	      index_t quB = 2,
	      bool verbose = false)
    {
	gsOptionList legendreOpts;
	legendreOpts.addInt ("quRule","Quadrature rule used (1) Gauss-Legendre; (2) Gauss-Lobatto; (3) Patch-Rule", gsQuadrature::GaussLegendre);
	legendreOpts.addReal("quA", "Number of quadrature points: quA*deg + quB", quA);
	legendreOpts.addInt ("quB", "Number of quadrature points: quA*deg + quB", quB);
	legendreOpts.addSwitch("overInt", "Apply over-integration or not?", false);
	gsQuadRule<real_t>::uPtr legendre = gsQuadrature::getPtr(spline.basis(), legendreOpts);

	gsMatrix<T> points;
	gsVector<T> weights;
	gsMatrix<T> values;
	T result = 0;

	if(verbose)
	    gsInfo << "Computing L2 error with a " << spline.degree(0)*quA + quB << "-point rule." << std::endl;

	for (auto domIt = spline.basis().makeDomainIterator(); domIt->good(); domIt->next() )
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
			    points, weights);

	    if(verbose)
		gsInfo << "The rule uses " << points.cols() << " points." << std::endl;

	    spline.eval_into(points, values);
	    for(index_t j=0; j<values.cols(); j++)
		result += weights(j) *
		    math::pow(values(0, j) - evalSample(points(0, j), points(1, j), sample), 2);
	}
	return math::sqrt(result);
    }

    template <class T>
    T L2Error(const gsGeometry<T>& spline,
	      index_t sample,
	      T quA = 2.0,
	      index_t quB = 2,
	      bool verbose = false)
    {
	auto cast = static_cast<const gsTensorBSpline<2, real_t>*>(&spline);
	return L2Error(*cast, sample, quA, quB, verbose);
    }
        
} // namespace gismo
