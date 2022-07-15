/** @file gsMatrixCrossApproximation_2.h

    @brief A matrix cross-approximation algorithm based on
    K. Frederix, M. Van Barel: Solving a large dense linear system by
    adaptive cross approximation, JCAM.

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

template <class T>
class gsMatrixCrossApproximation_2
{

public:

    gsMatrixCrossApproximation_2(const gsMatrix<T>& input)
	: m_input(input)
    {
    }

    void compute();

    inline void getU(gsMatrix<T>& result) const
    {
	result = m_U;
    }

    inline void getV(gsMatrix<T>& result) const
    {
	result = m_V;
    }

    inline void getT(gsMatrix<T>& result) const
    {
	result = m_T;
    }

protected:

    bool stopcrit();

    index_t chooseRow(const std::set<index_t>& usedIndices) const;

    gsMatrix<T> m_input;

    gsMatrix<T> m_U, m_V, m_T;


};

} // namespace gismo
