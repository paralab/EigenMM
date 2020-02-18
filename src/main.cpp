///////////////////////////////////////////////////////////////////////////////
//
// File main.cpp
//
// The MIT License
//
// Copyright (c) 2006 Division of Applied Mathematics, Brown University (USA),
// Department of Aeronautics, Imperial College London (UK), and Scientific
// Computing and Imaging Institute, University of Utah (USA).
//
// License for the specific language governing rights and limitations under
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
//
// Description: EigenMM example driver
//
///////////////////////////////////////////////////////////////////////////////

#include <eigen_mm.h>

void loadMatsFromFile(Mat *K, Mat *M, const char* operators_path)
{
    PetscViewer viewer;
    char K_filename[1024] = {};
    char M_filename[1024] = {};
    sprintf(K_filename, "%s_K", operators_path);
    sprintf(M_filename, "%s_M", operators_path);

    PetscViewerBinaryOpen(PETSC_COMM_WORLD, K_filename, FILE_MODE_READ, &viewer);
    MatCreate(PETSC_COMM_WORLD, K);
    MatLoad(*K, viewer);
    MatSetOption(*K, MAT_HERMITIAN, PETSC_TRUE);
    MatAssemblyBegin(*K, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(*K, MAT_FINAL_ASSEMBLY);
    PetscViewerDestroy(&viewer);

    PetscViewerBinaryOpen(PETSC_COMM_WORLD, M_filename, FILE_MODE_READ, &viewer);
    MatCreate(PETSC_COMM_WORLD, M);
    MatLoad(*M, viewer);
    MatSetOption(*M, MAT_HERMITIAN, PETSC_TRUE);
    MatAssemblyBegin(*M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(*M, MAT_FINAL_ASSEMBLY);
    PetscViewerDestroy(&viewer);
}

void setupOptions(SolverOptions &options, const char* optionsPath)
{
    TiXmlDocument doc;
    bool loadOkay = doc.LoadFile(optionsPath);

    if ( !loadOkay )
    {
        printf( "Could not load the options xml file. Error='%s'. Exiting.\n", doc.ErrorDesc() );
        exit(EXIT_FAILURE);
    }

    TiXmlHandle docH( &doc );
    TiXmlElement* element = docH.FirstChildElement( "EIGEN_MM" ).FirstChildElement( "OPTIONS" ).Element();

        options.set_splitmaxiters(atoi(element->Attribute( "_splitmaxiters" )));
        options.set_nodesperevaluator(atoi(element->Attribute( "_nodesperevaluator" )));
        options.set_subproblemsperevaluator(atoi(element->Attribute( "_subproblemsperevaluator" )));
        options.set_totalsubproblems(atoi(element->Attribute( "_totalsubproblems" )));
        options.set_nevaluators(atoi(element->Attribute( "_nevaluators" )));
        options.set_taskspernode(atoi(element->Attribute( "_taskspernode" )));
        options.set_nevals(atoi(element->Attribute( "_nevals" )));
        options.set_nk(atoi(element->Attribute( "_nk" )));
        options.set_nb(atoi(element->Attribute( "_nb" )));
        options.set_p(atoi(element->Attribute( "_p" )));
        options.set_nv(atoi(element->Attribute( "_nv" )));
        options.set_raditers(atoi(element->Attribute( "_raditers" )));
        options.set_splittol(atof(element->Attribute( "_splittol" )));
        options.set_radtol(atof(element->Attribute( "_radtol" )));
        options.set_L(atof(element->Attribute( "_L" )));
        options.set_R(atof(element->Attribute( "_R" )));
        options.set_terse(atoi(element->Attribute( "_terse" )));
        options.set_details(atoi(element->Attribute( "_details" )));
        options.set_debug(atoi(element->Attribute( "_debug" )));
        options.set_save_correctness(atoi(element->Attribute( "_save_correctness" )), element->Attribute("_correctness_filename"));
        options.set_save_operators(atoi(element->Attribute( "_save_operators" )), element->Attribute("_operators_filename"));
        options.set_save_eigenvalues(atoi(element->Attribute( "_save_eigenvalues" )), element->Attribute("_eigenvalues_filename"));
        options.set_save_eigenbasis(atoi(element->Attribute( "_save_eigenbasis" )), element->Attribute("_eigenbasis_filename"));

    MPI_Barrier(PETSC_COMM_WORLD);
}

int main(int argc, char *argv[])
{
    Mat K, M, V;
    Vec lambda;
    SolverOptions options;
    eigen_mm solver;

    const char* operators_path;
    const char* options_path;
    if (argc < 2)
    {
        operators_path = "data/operators/";
        options_path = "options.xml";
    }
    else if (argc < 3)
    {
        operators_path = argv[1];
        options_path = "options.xml";
    }
    else
    {
        operators_path = argv[1];
        options_path = argv[2];
    }
    
    // Initialize SLEPc
    SlepcInitialize(NULL, NULL, NULL, NULL);

    setupOptions(options, options_path);

    loadMatsFromFile(&K, &M, operators_path);

    PetscInt nrows, ncols;
    MatGetSize(K, &nrows, &ncols);
    PetscPrintf(PETSC_COMM_WORLD, "Size of K/M: %d by %d\n", nrows, ncols);

    PetscPrintf(PETSC_COMM_WORLD, "Operators path: %s\n", operators_path);

    solver.init(K, M, options);

    solver.solve(&V, &lambda);

    SlepcFinalize();
}
