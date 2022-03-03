/** @file gsWriteGnuplot.h

    @brief Exporting vectors in a format suitable for plotting with
    Gnuplot.

    This file is part of the G+Smo library.
    
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.

    Author(s): D. Mokris
*/

#pragma once

#include <fstream>
#include <vector>
#include <string>

#include <gsIO/gsIOUtils.h> // Basically only because of namespace gismo and GISMO_ASSERT.

namespace gismo
{

    template <class Tx, class Ty>
    void gsWriteGnuplot(const std::vector<Tx>& xData,
			const std::vector<Ty>& yData,
			const std::string& filename)
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
    void gsWriteGnuplot(const std::vector<T>& data, const std::string& filename)
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

} // namespace gismo
