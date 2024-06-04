#include <iostream>
#include <boost/program_options.hpp>
#include <stdio.h>
#include <chrono>
#include <cmath>
#include <cstring>
#include <omp.h>
#include <openacc.h>

#define OFFSET(x, y, m) (((x)*(m)) + (y))

using namespace std;
namespace opt = boost::program_options;

void initialize(double *A, double *Anew, int m, int n)
{
    memset(A, 0, n * m * sizeof(double));
    memset(Anew, 0, n * m * sizeof(double));

#pragma acc parallel loop independent  
    for(int i = 0; i < m; i++)
    {
        //(y0 * (x1 - x) + y1 * (x - x0)) / (x1 - x0)

        A[i] = 10 + (float)(10 * i) / (m-1);
        Anew[i] = 10 + (float)(10 * i) / (m-1);

        A[OFFSET(i, m-1, m)] = 20 + (float)(10 * i) / (n-1);
        Anew[OFFSET(i, m-1, m)] = 20 + (float)(10 * i) / (n-1);

        A[OFFSET(n-1, i, m)] = 20 + (float)(10 * i) / (m-1);
        Anew[OFFSET(n-1, i, m)] = 20 + (float)(10 * i) / (m-1);

        A[OFFSET(i, 0, m)] = 10 + (float)(10 * i) / (n-1);
        Anew[OFFSET(i, 0, m)] = 10 + (float)(10 * i) / (n-1);
    }
}

void deallocate(double *A, double *Anew)
{
    free(A);
    free(Anew);
}

int main(int argc, char *argv[])
{
    opt::options_description desc("All options");

    desc.add_options()
            ("precision", opt::value<double>())
            ("size", opt::value<int>())
            ("iter", opt::value<int>())
            ;

    opt::variables_map vm;
    opt::store(opt::parse_command_line(argc, argv, desc), vm);
    opt::notify(vm);
    const int n = vm["size"].as<int>();
    const int m = vm["size"].as<int>();
    const int iter_max = vm["iter"].as<int>();

    const double tol = vm["precision"].as<double>();

    double *A = (double*) malloc(n * m * sizeof(double));
    double *Anew = (double*) malloc(n * m * sizeof(double));

    #pragma acc enter data create(A[0:n*m], Anew[0:n*m])
    initialize(A, Anew, m, n);

    cout << "Jacobi relaxation Calculation: " << n << " x " << m << " mesh" << endl;

    auto start = std::chrono::steady_clock::now();
    int iter = 0;

    double err = 1.0;

    {
        while (err > tol && iter < iter_max)
        {
            err = 0.0;
            #pragma acc parallel loop collapse(2) reduction(max:err)
            for( int j = 1; j < n-1; j++)
            {
                for( int i = 1; i < m-1; i++ )
                {
                    Anew[OFFSET(j, i, m)] = 0.2 * (A[OFFSET(j, i, m)] + 
                                                   A[OFFSET(j, i+1, m)] + 
                                                   A[OFFSET(j, i-1, m)] + 
                                                   A[OFFSET(j-1, i, m)] + 
                                                   A[OFFSET(j+1, i, m)]);

                    err = fmax( err, fabs(Anew[OFFSET(j, i, m)] - A[OFFSET(j, i , m)]));
                }
            }

            #pragma acc parallel loop collapse(2)
            for( int j = 1; j < n-1; j++)
            {
                for( int i = 1; i < m-1; i++ )
                {
                    A[OFFSET(j, i, m)] = Anew[OFFSET(j, i, m)];
                }
            }

            iter++;
        }
    }

    #pragma acc exit data copyout(A[0:n*m]) delete(Anew[0:n*m])

    auto end = std::chrono::steady_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    cout << iter << ", " << err << endl;

    cout << "total: " << duration.count() << " mls" << endl;

    deallocate(A, Anew);

    return 0;
}
