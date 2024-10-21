#include <cassert>
#include <iostream>
#include <fstream>
#include <iomanip>

//////////////////////////////////////////////////////////////////////////
//Function Prototypes
//////////////////////////////////////////////////////////////////////////
void ReadMatrix(std::string name,int& noRows,int& noCols,double**& matrix,
    double*& rhsVector);
void PrintMatrix(int noRows,int noCols,double** matrix);
//////////////////////////////////////////////////////////////////////////

int main()
{
    double** matrix;
    double* rhs_vector;
    int no_rows, no_cols;

    //Read in the appropriate matrix
    ReadMatrix("matrix_vector3.dat",no_rows,no_cols,matrix,rhs_vector);
    
    /////////////////////
    
    //Backwards Gaussian Elimination (partial pivoting)
    for (int i=no_rows; i>=1; i--)
    {
        //Perform pivoting
        //Calculating k
        int k=0;
        for (int j=1; j<=i; j++)
        {
            if (abs(matrix[j-1][i-1]) > abs(matrix[k][i-1]))
            {
                k = j-1;
            }
        }

        //Pointers to k-th and i-th rows of A
        double* row_k = matrix[k];
        double* row_i = matrix[i-1];
        //Swap k-th and i-th rows
        matrix[k] = row_i;
        matrix[i-1] = row_k;

        //Values at k-th and i-th elements of b
        double element_k = rhs_vector[k];
        double element_i = rhs_vector[i-1];
        //Swap k-th and i-th elements of b
        rhs_vector[k] = element_i;
        rhs_vector[i-1] = element_k;

        //Scale current row
        rhs_vector[i-1] /= matrix[i-1][i-1];
        for (int m=1; m<=i; m++)
        {
            matrix[i-1][m-1] /= matrix[i-1][i-1];
        }

        //Perform Elimination
        for (int j=i-1; j>=1; j--)
        {
            rhs_vector[j-1] -= matrix[j-1][i-1] * rhs_vector[i-1];
            for (int m=1; m<=i-1; m++)
            {
                matrix[j-1][m-1] -= matrix[j-1][i-1] * matrix[i-1][m-1];
            }
        }

    }

    //Initialising the solution vector
    double* solution_vector;
    solution_vector = new double [no_rows];

    //Forward substituion
    for (int i=1; i<=no_rows; i++)
    {
        solution_vector[i-1] = rhs_vector[i-1];
        for (int j=1; j<=i-1; j++)
        {
            solution_vector[i-1] -= matrix[i-1][j-1]*solution_vector[j-1];
        }
    }

    //Printing the final solution vector
    std::cout << "The final solution vector is:" << std::endl;
    for (int i=1; i<=no_rows; i++)
    {
        std::cout << solution_vector[i-1] << "\n";
    }
    std::cout << std::endl;

    //Delete the solution vector
    delete [] solution_vector;

    /////////////////////

    //Example of printing the matrix to the screen
    PrintMatrix(no_rows,no_cols,matrix);


}

//////////////////////////////////////////////////////////////////////////
//Function Declarations
//////////////////////////////////////////////////////////////////////////

//Function to read matrix and rhs vector from file
void ReadMatrix(std::string name,int& noRows,int& noCols,double**& matrix,
    double*& rhsVector)
{
    // Open the file
    std::ifstream readFile(name);
    assert(readFile.is_open());

    readFile >> noRows >> noCols;

    //Allocate the matrix and read in
    matrix = new double*[noRows];
    rhsVector = new double[noRows];

    for (int i=0;i<noRows;i++)
    {
        matrix[i] = new double[noCols];
        for (int j=0;j<noCols;j++)
        {
            readFile >> matrix[i][j];
        }
    }

    //Allocate the RHS vector and read in
    for (int i=0;i<noRows;i++)
    {
        readFile >> rhsVector[i];
    }

    //Clean up
    readFile.close();
    return;
}

//Function to print matrix to the screen
void PrintMatrix(int noRows,int noCols,double** matrix)
{
    std::cout << "no_rows = " << noRows << ", no_cols = " << noCols << std::endl;

    for (int i=0; i < noRows; ++i)
    {
        for (int j=0; j < noCols; ++j)
        {
            if (j != 0) std::cout << " ";

            std::cout << std::setw(12) << std::setfill(' ') << std::setprecision(6) << matrix[i][j];
        }
        std::cout << std::endl;
    }
}