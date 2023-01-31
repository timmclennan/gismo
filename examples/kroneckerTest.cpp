/** @file kronecherTest.cpp

    @brief Test to see if the gsSparseMatrix kron function works

    This file is part of the G+Smo library.

    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.

    Author(s): J. Sogn
*/

#include <iostream>
#include <gismo.h>


using namespace gismo;


int main(int argc, char *argv[])
{
    

    gsSparseMatrix<real_t, ColMajor> C1(2,3);
    C1.reservePerColumn( 2 ); 
    C1(0,0) = -2.0; C1(0,1) = 0.5;  C1(0,2) = -3.0;
    C1(1,0) =  2.0; C1(1,1) = 5.0;  C1(1,2) =  1.0; 
    C1.makeCompressed();
    
    gsSparseMatrix<real_t, ColMajor> C2(3,2);
    C2.reservePerColumn( 3 ); 
    C2(0,0) = -2.0; C2(0,1) = 0.5;
    C2(1,0) =  3.0; C2(1,1) = 4.0;
    C2(2,0) = -1.0; C2(2,1) = 2.0;
    C2.makeCompressed();
    
    gsSparseMatrix<real_t, RowMajor> R1(2,3);
    R1.reservePerColumn( 3 ); //Row version does not exit
    R1(0,0) = -2.0; R1(0,1) = 0.5;  R1(0,2) = -3.0;
    R1(1,0) =  2.0; R1(1,1) = 5.0;  R1(1,2) =  1.0; 
    R1.makeCompressed();
        
    gsSparseMatrix<real_t, RowMajor> R2(3,2);
    R2.reservePerColumn( 2 );  //Row version does not exit
    R2(0,0) = -2.0; R2(0,1) = 0.5;
    R2(1,0) =  3.0; R2(1,1) = 4.0;
    R2(2,0) = -1.0; R2(2,1) = 2.0;
    R2.makeCompressed();
    
    gsInfo << "ColMajor: C1.kron(C2)\n" << C1.kron(C2) <<"\n";
    gsInfo << "RowMajor: R1.kron(R2)\n" << R1.kron(R2) <<"\n";

    
    return 0;
}
