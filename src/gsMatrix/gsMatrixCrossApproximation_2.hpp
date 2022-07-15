/** @file gsMatrixCrossApproximation_2.h

    @brief Bodies of functions from gsMatrixCrossApproximation_2.h

    This file is part of the G+Smo library.
    
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.

    Author(s): D. Mokris
*/

#include <gsMatrix/gsMatrixCrossApproximation_2.h>
#include <set>

namespace gismo
{

template <class T>
void gsMatrixCrossApproximation_2<T>::compute()
{
    std::set<index_t> usedIndices; // Z_i
    index_t p = 0;
    index_t row = chooseRow(usedIndices);
    while(!stopcrit())
    {
	bool found = false;
	while(!found)
	{
	    // Compute the maximal entry in modulus of the row.
	    m_T(p, p) = findAbsMax(row);
	    found = (math::abs(m_T(p, p)) > tol);
	    usedIndices.insert(row);

	    if(!found)
	    {
		if(usedIndices.size() != m)
		    row = chooseRow(usedIndices);
		else
		    stopcrit = true;
	    }
	}
	
	    
}

} // namespace gismo
