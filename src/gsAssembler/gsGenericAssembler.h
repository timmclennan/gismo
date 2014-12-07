/** @file gsGenericAssembler.h

    @brief Provides an assembler for common IGA matrices

    This file is part of the G+Smo library.

    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.

    Author(s): S. Kleiss, A. Mantzaflaris
*/

#pragma once

#include <gsAssembler/gsAssemblerBase.h>
#include <gsAssembler/gsVisitorMass.h>
#include <gsAssembler/gsVisitorGradGrad.h>

namespace gismo
{

/**
   @brief Assembles the mass, stiffness matrix on a given domain

 */
template <class T>
class gsGenericAssembler : public gsAssemblerBase<T>
{
public:
    typedef gsAssemblerBase<T> Base;

public:

    /// Constructor with gsMultiBasis
    gsGenericAssembler( gsMultiPatch<T> const         & patches,
                        gsMultiBasis<T> const         & bases,
                        bool conforming = false)
    : Base(patches)
    {
        m_bases.push_back(bases);

        // Init mapper
        m_dofMappers.resize(1);
        bases.getMapper(conforming, m_dofMappers.front() );
        m_dofs = m_dofMappers.front().freeSize();
    }

    /// Mass assembly routine
    const gsSparseMatrix<T> & assembleMass()
    {
        // Pre-allocate non-zero elements for each column of the
        // sparse matrix
        int nonZerosPerCol = 1;
        for (int i = 0; i < m_bases.front().dim(); ++i) // to do: improve
            nonZerosPerCol *= 2 * m_bases.front().maxDegree(i) + 1;
        m_matrix.resize(m_dofs, m_dofs); // Clean matrices
        m_matrix.reserve( gsVector<int>::Constant(m_dofs, nonZerosPerCol) );

        // Mass visitor
        gsVisitorMass<T> mass;

        for (unsigned np=0; np < m_patches.nPatches(); ++np )
        {
            //Assemble mass matrix and rhs for the local patch
            // with index np and add to m_matrix
            this->apply(mass, np);
        }

        // Assembly is done, compress the matrix
        m_matrix.makeCompressed();   
        return m_matrix;
    }

    /// Stiffness assembly routine
    const gsSparseMatrix<T> & assembleStiffness()
    {
        // Pre-allocate non-zero elements for each column of the
        // sparse matrix
        int nonZerosPerCol = 1;
        for (int i = 0; i < m_bases.front().dim(); ++i) // to do: improve
            nonZerosPerCol *= 2 * m_bases.front().maxDegree(i) + 1;
        m_matrix.resize(m_dofs, m_dofs); // Clean matrices
        m_matrix.reserve( gsVector<int>::Constant(m_dofs, nonZerosPerCol) );

        // Stiffness visitor
        gsVisitorGradGrad<T> stiffness;

        for (unsigned np=0; np < m_patches.nPatches(); ++np )
        {
            //Assemble stiffness matrix and rhs for the local patch
            // with index np and add to m_matrix
            this->apply(stiffness, np);
        }

        // Assembly is done, compress the matrix
        m_matrix.makeCompressed();   
        return m_matrix;
    }

    /// Stiffness assembly routine on patch \a patchIndex
    const gsSparseMatrix<T> & assembleMass(int patchIndex)
    {
        const int sz = m_bases.front()[patchIndex].size();

        // Pre-allocate non-zero elements for each column of the
        // sparse matrix
        int nonZerosPerCol = 1;
        for (int i = 0; i < m_bases.front().dim(); ++i) // to do: improve
            nonZerosPerCol *= 2 * m_bases.front()[patchIndex].degree(i) + 1;
        m_matrix.resize(sz, sz); // Clean matrix
        m_matrix.reserve( gsVector<int>::Constant(sz, nonZerosPerCol) );

        // Mass visitor (without mapper)
        gsVisitorMass<T> mass(false);
        
        //Assemble stiffness matrix for this patch
        this->apply(mass, patchIndex);

        // Assembly is done, compress the matrix
        m_matrix.makeCompressed();   
        return m_matrix;
    }

    /// Stiffness assembly routine on patch \a patchIndex
    const gsSparseMatrix<T> & assembleStiffness(int patchIndex)
    {
        const int sz = m_bases.front()[patchIndex].size();

        // Pre-allocate non-zero elements for each column of the
        // sparse matrix
        int nonZerosPerCol = 1;
        for (int i = 0; i < m_bases.front().dim(); ++i) // to do: improve
            nonZerosPerCol *= 2 * m_bases.front()[patchIndex].degree(i) + 1;
        m_matrix.resize(sz, sz); // Clean matrix
        m_matrix.reserve( gsVector<int>::Constant(sz, nonZerosPerCol) );

        // Stiffness visitor (without mapper)
        gsVisitorGradGrad<T> stiffness(false);

        //Assemble stiffness matrix for this patch
        this->apply(stiffness, patchIndex);

        // Assembly is done, compress the matrix
        m_matrix.makeCompressed();   
        return m_matrix;
    }
    
    // special member function for anyone who hates lower triangular matrices
    Eigen::SparseSelfAdjointView< typename gsSparseMatrix<T>::Base, Lower> fullMatrix()
    {
        return m_matrix.template selfadjointView<Lower>();
    }
    

private:

    // Members from gsAssemblerBase
    using gsAssemblerBase<T>::m_patches;
    using gsAssemblerBase<T>::m_bases;
    using gsAssemblerBase<T>::m_dofMappers;
    using gsAssemblerBase<T>::m_dofs;
    using gsAssemblerBase<T>::m_matrix;

private:
    // Hiding the rhs
    const gsMatrix<T> & rightHandSide() const;

};



} // namespace gismo

