/** @file ieti_example.cpp

    @brief Provides examples for the ieti solver.

    This file is part of the G+Smo library.

    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.

    Author(s): S. Takacs
*/

#include <gismo.h>
#include <gsIeti/gsIetiMapper.h>

using namespace gismo;

int main(int argc, char *argv[])
{
    /************** Define command line options *************/

    std::string geometry("domain2d/yeti_mp2.xml");
    index_t splitPatches = 1;
    real_t stretchGeometry = 1;
    index_t refinements = 1;
    index_t degree = 2;
    real_t tolerance = 1.e-8;
    index_t maxIterations = 100;
    bool plot = false;
    bool sol = false;
    std::string boundaryConditions("d");

    gsCmdLine cmd("Solves a PDE with an isogeometric discretization using a multigrid solver.");
    cmd.addString("g", "Geometry",              "Geometry file", geometry);
    cmd.addInt   ("",  "SplitPatches",          "Split every patch that many times in 2^d patches", splitPatches);
    cmd.addReal  ("",  "StretchGeometry",       "Stretch geometry in x-direction by the given factor", stretchGeometry);
    cmd.addInt   ("r", "Refinements",           "Number of uniform h-refinement steps to perform before solving", refinements);
    cmd.addInt   ("p", "Degree",                "Degree of the B-spline discretization space", degree);
    cmd.addReal  ("t", "Solver.Tolerance",      "Stopping criterion for linear solver", tolerance);
    cmd.addInt   ("",  "Solver.MaxIterations",  "Stopping criterion for linear solver", maxIterations);
    cmd.addString("b", "BoundaryConditions",    "Boundary conditions", boundaryConditions);
    cmd.addSwitch(     "sol",                   "Write the computed solution to console", sol);
    cmd.addSwitch(     "plot",                  "Plot the result with Paraview", plot);

    try { cmd.getValues(argc,argv); } catch (int rv) { return rv; }

    gsOptionList opt = cmd.getOptionList();

    if ( ! gsFileManager::fileExists(geometry) )
    {
        gsInfo << "Geometry file could not be found.\n";
        gsInfo << "I was searching in the current directory and in: " << gsFileManager::getSearchPaths() << "\n";
        return EXIT_FAILURE;
    }

    gsInfo << "Run ieti_example with options:\n" << opt << std::endl;

    /******************* Define geometry ********************/

    gsInfo << "Define geometry... " << std::flush;

    gsMultiPatch<>::uPtr mpPtr = gsReadFile<>(geometry);
    if (!mpPtr)
    {
        gsInfo << "No geometry found in file " << geometry << ".\n";
        return EXIT_FAILURE;
    }
    gsMultiPatch<>& mp = *mpPtr;

    for (index_t i=0; i<splitPatches; ++i)
    {
        gsInfo << "split patches uniformly... " << std::flush;
        mp = mp.uniformSplit();
    }

    if (stretchGeometry!=1)
    {
       gsInfo << "and stretch it... " << std::flush;
       for (size_t i=0; i!=mp.nPatches(); ++i)
           const_cast<gsGeometry<>&>(mp[i]).scale(stretchGeometry,0);
       // Const cast is allowed since the object itself is not const. Stretching the
       // overall domain keeps its topology.
    }

    gsInfo << "done.\n";

    /************** Define boundary conditions **************/

    gsInfo << "Define right-hand-side and boundary conditions... " << std::flush;

    // Right-hand-side
    gsFunctionExpr<> f( "2*sin(x)*cos(y)", mp.geoDim() );

    // Dirichlet function
    gsFunctionExpr<> gD( "sin(x)*cos(y)", mp.geoDim() );

    // Neumann
    gsConstantFunction<> gN( 1.0, mp.geoDim() );

    gsBoundaryConditions<> bc;
    {
        const index_t len = boundaryConditions.length();
        index_t i = 0;
        for (gsMultiPatch<>::const_biterator it = mp.bBegin(); it < mp.bEnd(); ++it)
        {
            char b_local;
            if ( len == 1 )
                b_local = boundaryConditions[0];
            else if ( i < len )
                b_local = boundaryConditions[i];
            else
            {
                gsInfo << "\nNot enough boundary conditions given.\n";
                return EXIT_FAILURE;
            }

            if ( b_local == 'd' )
                bc.addCondition( *it, condition_type::dirichlet, &gD );
            else if ( b_local == 'n' )
                bc.addCondition( *it, condition_type::neumann, &gN );
            else
            {
                gsInfo << "\nInvalid boundary condition given; only 'd' (Dirichlet) and 'n' (Neumann) are supported.\n";
                return EXIT_FAILURE;
            }

            ++i;
        }
        if ( len > i )
            gsInfo << "\nToo much boundary conditions have been specified. Ingnoring the remaining ones.\n";
        gsInfo << "done. "<<i<<" boundary conditions set.\n";
    }


    /************ Setup bases and adjust degree *************/

    gsMultiBasis<> mb(mp);

    gsInfo << "Setup bases and adjust degree... " << std::flush;

    for ( size_t i = 0; i < mb.nBases(); ++ i )
        mb[i].setDegreePreservingMultiplicity(degree);

    for ( index_t i = 0; i < refinements; ++i )
        mb.uniformRefine();

    gsInfo << "done.\n";

    /********* Setup assembler and assemble matrix **********/

    gsInfo << "Setup assembler and assemble matrix... " << std::flush;

    const index_t nPatches = mp.nPatches();

    gsIetiMapper<> ietiMapper;

    // We start by setting up a global FeSpace that allows us to
    // obtain a dof mapper and the Dirichlet data
    {
        typedef gsExprAssembler<>::space  space;
        gsExprAssembler<> assembler;
        space u = assembler.getSpace(mb);
        bc.setGeoMap(mp);
        u.setup(bc, dirichlet::interpolation, 0);
        ietiMapper.init( mb, u.mapper(), u.fixedPart() );
    }

    // Compute the jump matrices
    ietiMapper.computeJumpMatrices(true, true);

    // We tell the ieti mapper which primal constraints we want; calling
    // more than one such function is possible.
    ietiMapper.cornersAsPrimals();

    gsIetiSystem<> ieti;
    ieti.reserve(nPatches+1);

    gsScaledDirichletPrec<> prec;
    prec.reserve(nPatches);

    gsPrimalSystem<> primal;
    primal.init(ietiMapper.nPrimalDofs());

    for (index_t k=0; k<nPatches; ++k)
    {
        // We use the local variants of everything
        gsBoundaryConditions<> bc_local;
        bc.getConditionsForPatch(k,bc_local);
        gsMultiPatch<> mp_local = mp[k];
        gsMultiBasis<> mb_local = mb[k];

        // The usual stuff for the expression assembler
        typedef gsExprAssembler<>::geometryMap geometryMap;
        typedef gsExprAssembler<>::variable    variable;
        typedef gsExprAssembler<>::space       space;
        typedef gsExprAssembler<>::solution    solution;

        // We set up the assembler
        gsExprAssembler<> assembler(1,1);

        // Elements used for numerical integration
        assembler.setIntegrationElements(mb_local);
        gsExprEvaluator<> ev(assembler);

        // Set the geometry map
        geometryMap G = assembler.getMap(mp_local);

        // Set the discretization space
        space u = assembler.getSpace(mb_local);

        // Incorporate Dirichlet BC
        bc_local.setGeoMap(mp_local);
        u.setup(bc_local, dirichlet::interpolation, 0);

        // This function provides a new dof mapper and the Dirichlet data
        // This is necessary since it might happen that a 2d-patch touches the
        // Dirichlet boundary just with a corner or that a 3d-patch touches the
        // Dirichlet boundary with a corner or an edge. These cases are not
        // covered by bc.getConditionsForPatch
        ietiMapper.initFeSpace(u,k);

        // Set the source term
        variable ff = assembler.getCoeff(f, G);

        // Initialize the system
        assembler.initSystem(false);

        // Compute the system matrix and right-hand side
        assembler.assemble( igrad(u, G) * igrad(u, G).tr() * meas(G), u * ff * meas(G) );

        // Add contributions from Neumann conditions to right-hand side
        variable g_N = assembler.getBdrFunction();
        assembler.assembleRhsBc(u * g_N.val() * nv(G).norm(), bc.neumannSides() );

        // Fetch data
        gsSparseMatrix<real_t, RowMajor> jumpMatrix  = ietiMapper.jumpMatrix(k);
        gsSparseMatrix<>                 localMatrix = assembler.matrix();
        gsMatrix<>                       localRhs    = assembler.rhs();

        // Add the patch to the scaled Dirichlet preconditioner
        prec.addSubdomain(
            gsScaledDirichletPrec<>::restrictToSkeleton(
                jumpMatrix,
                localMatrix,
                ietiMapper.getSkeletonDofs(k)
            )
        );

        // This function writes back to jumpMatrix, localMatrix, and localRhs,
        // so it must be called after prec.addSubdomain().
        primal.handleConstraints(
            ietiMapper.primalConstraints(k),
            ietiMapper.primalDofIndices(k),
            jumpMatrix,
            localMatrix,
            localRhs
        );

        // Add the patch to the Ieti system
        ieti.addSubdomain(
            jumpMatrix.moveToPtr(),
            makeMatrixOp(localMatrix.moveToPtr()),
            give(localRhs)
        );
    }

    // Add the primal problem if there are primal constraints
    if (ietiMapper.nPrimalDofs()>0)
        ieti.addSubdomain(
            primal.jumpMatrix().moveToPtr(),
            makeMatrixOp(primal.localMatrix().moveToPtr()),
            give(primal.localRhs())
        );

    gsInfo << "done.\n";

    /**************** Setup solver and solve ****************/

    gsInfo << "Setup solver and solve... \n"
        "    Setup multiplicity scaling... " << std::flush;

    // Tell the preconditioner to set up the scaling
    prec.setupMultiplicityScaling();

    gsInfo << "done.\n    Setup rhs... " << std::flush;
    // Compute the Schur-complement contribution for the right-hand-side
    gsMatrix<> rhsForSchur = ieti.rhsForSchurComplement();

    gsInfo << "done.\n    Setup cg solver for Lagrange multipliers and solve... " << std::flush;
    // Initial guess
    gsMatrix<> lambda;
    lambda.setRandom( ieti.numberOfLagrangeMultipliers(), 1 );

    // This is the main cg iteration
    gsMatrix<> errorHistory;
    gsConjugateGradient<>( ieti.schurComplement(), prec.preconditioner() )
        .setOptions( opt.getGroup("Solver") )
        .solveDetailed( rhsForSchur, lambda, errorHistory );

    gsInfo << "done.\n    Reconstruct solution from Lagrange multipliers... " << std::flush;
    // Now, we want to have the global solution for u
    gsMatrix<> uVec = ietiMapper.constructGlobalSolutionFromLocalSolutions(
        primal.distributePrimalSolution(
            ieti.constructSolutionFromLagrangeMultipliers(lambda)
        )
    );
    gsInfo << "done.\n\n";

    /******************** Print end Exit ********************/

    const index_t iter = errorHistory.rows()-1;
    const bool success = errorHistory(iter,0) < tolerance;
    if (success)
        gsInfo << "Reached desired tolerance after " << iter << " iterations:\n";
    else
        gsInfo << "Did not reach desired tolerance after " << iter << " iterations:\n";

    if (errorHistory.rows() < 20)
        gsInfo << errorHistory.transpose() << "\n\n";
    else
        gsInfo << errorHistory.topRows(5).transpose() << " ... " << errorHistory.bottomRows(5).transpose()  << "\n\n";

    if (sol)
        gsInfo << uVec.transpose() << "\n\n";

    if (plot)
    {
        gsInfo << "Plotting in Paraview.\n";
        // Construct the solution as a scalar field
        // For this purpose, we use a global assembler
        gsExprAssembler<> A(1,1);
        //A.setOptions(Aopt);
        typedef gsExprAssembler<>::geometryMap geometryMap;
        typedef gsExprAssembler<>::variable    variable;
        typedef gsExprAssembler<>::space       space;
        typedef gsExprAssembler<>::solution    solution;
        // Elements used for numerical integration
        A.setIntegrationElements(mb);
        gsExprEvaluator<> ev(A);
        // Set the geometry map
        geometryMap G = A.getMap(mp);
        // Set the discretization space
        space u = A.getSpace(mb);
        // Solution vector and solution variable
        solution u_sol = A.getSolution(u, uVec);
        // Setup u
        u.setup(bc, dirichlet::interpolation, 0);
        ev.options().setSwitch( "plot.elements", true );
        ev.writeParaview( u_sol, G, "ieti_result" );
        //gsFileManager::open("ieti_result.pvd");
    }
    else
    {
        gsInfo << "Done. No output created, re-run with --plot to get a ParaView "
                  "file containing the solution.\n";
    }
    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}