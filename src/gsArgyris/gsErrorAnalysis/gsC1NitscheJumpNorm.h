/** @file gsC1NitscheJumpNorm.h

    @brief Computes the H1 norm with the jump; for approx. g1 Basis.

    This file is part of the G+Smo library.

    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.

    Author(s): P. Weinmüller
*/


#pragma once

namespace gismo
{

/** @brief The gsSeminormH1 class provides the functionality
 * to calculate the H1 - seminorm between a field and a function.
 *
 * \ingroup Assembler
*/
    template <class T>
    class gsC1NitscheJumpNorm
    {

    public:

        gsC1NitscheJumpNorm(const gsMultiPatch<T> & multiPatch,
                            const gsMultiPatch<T> & discretSolution,
                            const gsFunctionWithDerivatives<T> &exactSolution)
                : patchesPtr( &multiPatch ), discretSol( &discretSolution ), exactSol(exactSolution)
        {
            m_value.resize(patchesPtr->interfaces().size());
            m_value.setZero();
            m_valueSum = T(0.0);
        }


    public:

        void compute(bool storeElWise = false)
        {

            for (size_t numInt = 0; numInt < patchesPtr->interfaces().size(); numInt++ )
            {
                T value = T(0.0);

                gsMatrix<T> quNodes_L; // Temp variable for mapped nodes
                gsVector<T> quWeights; // Temp variable for mapped weights
                gsQuadRule<T> QuRule_L; // Reference Quadrature rule

                gsMatrix<T> quNodes_R; // Temp variable for mapped nodes
                gsQuadRule<T> QuRule_R; // Reference Quadrature rule

                // Evaluation flags for the Geometry map
                unsigned evFlags(0);

                const boundaryInterface & iFace = patchesPtr->interfaces()[numInt];

                size_t L = iFace.second().patch;
                size_t R = iFace.first().patch;

                std::vector<size_t> numInt_vector;
                numInt_vector.push_back(numInt);

                // Obtain an integration domain
                const gsBasis<T> & dom_L = discretSol->basis(L);
                const gsBasis<T> & dom_R = discretSol->basis(R);

                boxSide side_L = iFace.second();
                boxSide side_R = iFace.first();

                // Initialize visitor
                initializeb(dom_L, QuRule_L, evFlags, side_L);
                initializeb(dom_R, QuRule_R, evFlags, side_R);

                // Initialize geometry evaluator
                const gsGeometry<T> & patch_L = patchesPtr->patch(L);
                const gsGeometry<T> & patch_R = patchesPtr->patch(R);

                const gsAffineFunction<> ifaceMap(patchesPtr->getMapForInterface(iFace, dom_R.size() >= dom_L.size() ? 1 : -1));

                typename gsBasis<T>::domainIter domIt = dom_R.size() >= dom_L.size() ? dom_R.makeDomainIterator(side_R) : dom_L.makeDomainIterator(side_L);
                for (; domIt->good(); domIt->next())
                {
                    // Map the Quadrature rule to the element
                    gsMatrix<T> domItCorner(2,2);
                    domItCorner.col(0) = domIt->lowerCorner();
                    domItCorner.col(1) = domIt->upperCorner();

                    dom_R.size() >= dom_L.size() ? QuRule_R.mapTo(domIt->lowerCorner(), domIt->upperCorner(), quNodes_R, quWeights) :
                    QuRule_L.mapTo(domItCorner.col(0), domItCorner.col(1), quNodes_L, quWeights);

                    ifaceMap.eval_into(domItCorner,domItCorner);
                    /*
                    if (domItCorner(1-side_L.direction(),0) > domItCorner(1-side_L.direction(),1) && dom_R.size() >= dom_L.size()) // integral border switched
                    {
                        gsMatrix<T> temp_domItCorner = domItCorner;
                        domItCorner.col(1) = temp_domItCorner.col(0);
                        domItCorner.col(0) = temp_domItCorner.col(1);
                    }
                    else if (domItCorner(1-side_R.direction(),0) > domItCorner(1-side_R.direction(),1) && dom_R.size() < dom_L.size()) // integral border switched
                    {
                        gsMatrix<T> temp_domItCorner = domItCorner;
                        domItCorner.col(1) = temp_domItCorner.col(0);
                        domItCorner.col(0) = temp_domItCorner.col(1);
                    }
                    */

                    dom_R.size() >= dom_L.size() ? QuRule_L.mapTo(domItCorner.col(0), domItCorner.col(1), quNodes_L, quWeights):
                    QuRule_R.mapTo(domItCorner.col(0), domItCorner.col(1), quNodes_R, quWeights);

                    quWeights = quWeights.cwiseAbs(); // if at the interface the direction is not the same

                    bool switch_side = false;
                    if (side_L.direction() != side_R.direction())
                        switch_side = true;

                    // Evaluate on quadrature points
                    evaluateb(patch_L, quNodes_L, patch_R, quNodes_R, discretSol, exactSol);

                    // Accumulate value from the current element (squared)
                    computeb(side_L, quWeights, value, switch_side);

                }
                m_valueSum += value;
                m_value(numInt) = takeRoot(value);
            }
            m_valueSum = takeRoot(m_valueSum);
        }


        gsVector<T> value() const { return m_value; }
        T valueSum() const { return m_valueSum; }

    protected:

        void initializeb(const gsBasis<T> & basis,
                         gsQuadRule<T> & rule,
                         unsigned      & evFlags,
                         boxSide side) // replace with geoEval ?
        {
            // Setup Quadrature
            const unsigned d = basis.dim();
            const int dir = side.direction();
            gsVector<index_t> numQuadNodes( d );
            for (unsigned i = 0; i < d; ++i)
                numQuadNodes[i] = basis.degree(i) + 1;
            numQuadNodes[dir] = 1;

            // Setup Quadrature
            rule = gsGaussRule<T>(numQuadNodes);// harmless slicing occurs here

            // Set Geometry evaluation flags
            md_L.flags = NEED_VALUE|NEED_MEASURE|NEED_JACOBIAN|NEED_GRAD_TRANSFORM;
            md_R.flags = NEED_VALUE|NEED_MEASURE|NEED_JACOBIAN|NEED_GRAD_TRANSFORM;
        }

        // Evaluate on element.
        void evaluateb(const gsGeometry<T> & patch_L,
                       gsMatrix<T> & quNodes_L,
                       const gsGeometry<T> & patch_R,
                       gsMatrix<T> & quNodes_R,
                       const gsMultiPatch<T> * discretSol,
                       const gsFunctionWithDerivatives<T> & exactSol)
        {

            index_t patchIndex_L = patch_L.id();
            index_t patchIndex_R = patch_R.id();

            md_L.points = quNodes_L;
            md_R.points = quNodes_R;

            std::vector<gsMatrix<T>> basisData_L, basisData_R;
            discretSol->patch(patchIndex_L).evalAllDers_into(quNodes_L, 1, basisData_L);
            discretSol->patch(patchIndex_R).evalAllDers_into(quNodes_R, 1, basisData_R);

            // Sum up the row-wise solution
            f1ders.setZero(2, quNodes_L.cols());
            for (index_t i = 0; i < basisData_L[0].rows(); i++)
                f1ders += basisData_L[1].block(2*i,0,2,quNodes_L.cols());

            patch_L.computeMap(md_L);

            // Evaluate second function
            f2ders.setZero(2, quNodes_R.cols());
            for (index_t i = 0; i < basisData_R[0].rows(); i++)
                f2ders += basisData_R[1].block(2*i,0,2,quNodes_R.cols());

            patch_R.computeMap(md_R);
        }

        // assemble on element
        inline T computeb(boxSide side,
                          gsVector<T> const      & quWeights,
                          T & accumulated,
                          bool switched_side)
        {

            T sum(0.0);
            for (index_t k = 0; k < quWeights.rows(); ++k) // loop over quadrature nodes
            {
                //const T d = element.dim();
                // Transform the gradients
                transformGradients(md_L, k, f1ders, f1pders);
                transformGradients(md_R, k, f2ders, f2pders);

                // Compute the unit normal
                gsVector<T> unormal;

                outerNormal(md_L, k, side, unormal);

                const T weight = quWeights[k] * unormal.norm() ;

                // f2ders : N X 1
                sum += weight * ( (f1pders - f2pders).transpose() * unormal ).squaredNorm() ;

                //sum += weight * ( (f1ders - f2ders).transpose() * normal ).squaredNorm() ;
            }
            accumulated += sum;

            return sum;
        }

        inline T takeRoot(const T v) { return math::sqrt(v);}



    protected:

        const gsMultiPatch<T> * patchesPtr;

        const gsMultiPatch<T> * discretSol;

        const gsFunctionWithDerivatives<T> & exactSol;

    protected:
        gsVector<T> m_value;     // the total value of the norm
        T m_valueSum;
    protected:
        gsMatrix<T> f1pders, f2pders; // f2pders only needed if f2param = true
        gsMatrix<T> f1ders, f2ders;
        gsMapData<T> md_L, md_R;

    };


} // namespace gismo