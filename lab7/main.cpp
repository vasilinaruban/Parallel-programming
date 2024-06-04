#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <boost/program_options.hpp>
#include "nvtx3/nvToolsExt.h"
#include <openacc.h>
#include <cublas_v2.h>
#define OFFSET(x, y, m) (((x)*(m)) + (y))
namespace opt = boost::program_options;

void PrintMatrix(const double *A, size_t n) {
#pragma acc update host(A[0:n*n])
    for (size_t y = 0; y < n; ++y) {
        for (size_t x = 0; x < n; ++x) {
            std::cout << A[y * n + x] << " ";
        }
        std::cout << std::endl;
    }
}

int ProgramOptions(int argc, char **argv, double &epsilon, size_t &n, size_t &max_iter) {
    opt::options_description desc("опции");
    desc.add_options()
        ("epsilon, e", opt::value<double>()->default_value(1e-6), "precision")
        ("size, s", opt::value<size_t>()->default_value(256), "matrix size")
        ("iter, i", opt::value<size_t>()->default_value(1000000), "maximum nuber of iterations")
        ("help, h", "help");

    opt::variables_map vm;
    opt::store(opt::parse_command_line(argc, argv, desc), vm);
    opt::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 1;
    }
    
    epsilon = vm["epsilon"].as<double>();
    n = vm["size"].as<size_t>();
    max_iter = vm["iter"].as<size_t>();
    
    return 0;
}

void initialize(double *A, double *new_A, int m, int n)
{
    memset(A, 0, n * m * sizeof(double));
    memset(new_A, 0, n * m * sizeof(double));

    double delta = 10.0;

#pragma acc parallel loop independent  
    for(int i = 0; i < m; i++)
    {
        //(y0 * (x1 - x) + y1 * (x - x0)) / (x1 - x0)

        A[i] = 10 + (delta * i) / (m - 1);
        A[OFFSET(i, m - 1, m)] = 20 + (delta * i) / (n - 1);
        A[OFFSET(n - 1, i, m)] = 20 + (delta * i) / (m - 1);
        A[OFFSET(i, 0, m)] = 10 + (delta * i) / (n - 1);

        new_A[i] = 10 + (delta * i) / (m - 1);
        new_A[OFFSET(i, m - 1, m)] = 20 + (delta * i) / (n - 1);
        new_A[OFFSET(n - 1, i, m)] = 20 + (delta * i) / (m - 1);
        new_A[OFFSET(i, 0, m)] = 10 + (delta * i) / (n - 1);


    }
}

void Deallocate(double *A, double *new_A) {
#pragma acc exit data delete(A, new_A)
    delete[] A;
    delete[] new_A;
}

int main(int argc, char **argv) {
    nvtxRangePushA("init");
    double epsilon{};
    size_t n{};
    size_t max_iter{};
    if (ProgramOptions(argc, argv, epsilon, n, max_iter) != 0) {
        return 0;
    }
    
    int iter = 0;
    auto *A = new double[n * n];
    auto *new_A = new double[n * n];
    initialize(A, new_A, n, n);
    nvtxRangePop();

    double error = 1;
    auto const start = std::chrono::steady_clock::now();
    cublasStatus_t handle;
    cublasHandle_t handle;
    handle = cublasCreate(&handle);
    
    nvtxRangePushA("loop");
    double c = -1.0;
    int index = 0;
#pragma acc data copyin(index, c, A[0:n*n], new_A[0:n*n])
    {
    while (iter < max_iter && error > epsilon) 
    {
#pragma acc parallel loop independent collapse(2)
        for (size_t i = 1; i < n - 1; ++i) {
            for (size_t j = 1; j < n - 1; ++j) {
                new_A[OFFSET(j, i, m)] = 0.2 * (A[OFFSET(j, i, m)] + 
                                                   A[OFFSET(j, i+1, m)] + 
                                                   A[OFFSET(j, i-1, m)] +
                                                   A[OFFSET(j-1, i, m)] + 
                                                   A[OFFSET(j+1, i, m)]);
            }
        }

#pragma acc data present(A, new_A)
#pragma acc host_data use_device(new_A, A)
        {
            handle = cublasDaxpy(handle, n * n, &c, new_A, 1, A, 1);
            handle = cublasIdamax(handle, n * n, A, 1, &index);
        }


#pragma acc update self(A[index - 1])
        error = fabs(A[index - 1]);
#pragma acc host_data use_device(new_A, A)

        {
            handle = cublasDcopy(handle, n * n, new_A, 1, A, 1);
        }

        std::swap(A, new_A);

        if (iter % 1000 == 0)
        {
            std::cout << "iteration: " << iter + 1 << " error: " << error << std::endl;
        }
        iter++;
    }

    cublasDestroy(handle);
#pragma acc update self(new_A[0:n*n])
    }
    nvtxRangePop();
    auto const end = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed_seconds = end - start;
    if (n <= 13)
        PrintMatrix(A, n);
    std::cout << iter << "\n" << elapsed_seconds.count() << std::endl;
    Deallocate(A, new_A);
}
