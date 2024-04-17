#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>


double cpuSecond()
{
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ((double)ts.tv_sec + (double)ts.tv_nsec * 1.e-9);
}

/*
 * matrix_vector_product: Compute matrix-vector product c[m] = a[m][n] * b[n]
 */
void matrix_vector_product(double *a, double *b, double *c, size_t m, size_t n)
{
    double sum = 0.0;
    for (int i = 0; i < m; i++)
    {
        c[i] = 0.0;
        for (int j = 0; j < n; j++)
            c[i] += a[i * n + j] * b[j];
            sum += c[i];
    }
    //printf("serial: %lf\n", sum);
}

int bound (int id, int remainder, int items_per_thread)
{
    int bound = 0;
    if (id < remainder)
    {
        bound = id * (items_per_thread + 1);
    }
    else
    {
        bound = remainder * (items_per_thread + 1) + (id - remainder) * items_per_thread;
    }
    return bound;
}

void matrix_vector_product_omp(double *a, double *b, double *c, size_t m, size_t n)
{
    double sum = 0.0;
#pragma omp parallel
    {
        int nthreads = omp_get_num_threads();
        int threadid = omp_get_thread_num();
        int k = m % nthreads;
        int items_per_thread = m / nthreads;
        int lb = bound(threadid, k, items_per_thread);
        int ub = bound(threadid + 1, k, items_per_thread);

        double local_sum = 0.0;
        for (int i = lb; i < ub; i++)
        {
            c[i] = 0.0;
            for (int j = 0; j < n; j++) {
                c[i] += a[i * n + j] * b[j];
            }
            local_sum += c[i];
        }
#pragma omp atomic
        sum += local_sum;

    }
    //printf("parallel: %lf\n", sum);
}

void run_serial(size_t n, size_t m)
{
    double *a, *b, *c;
    std::unique_ptr<double[]> a(new double[m * n]);
    std::unique_ptr<double[]> b(new double[n]);
    std::unique_ptr<double[]> c(new double[m]);

    if (a == NULL || b == NULL || c == NULL)
    {
        printf("Error allocate memory!\n");
        exit(1);
    }

    for (size_t i = 0; i < m; i++)
    {
        for (size_t j = 0; j < n; j++)
            a[i * n + j] = i + j;
    }

    for (size_t j = 0; j < n; j++)
        b[j] = j;

    double t = cpuSecond();
    matrix_vector_product(a, b, c, m, n);
    t = cpuSecond() - t;

    printf("Elapsed time (serial): %.6f sec.\n", t);
}

void run_parallel(size_t n, size_t m)
{
    double *a, *b, *c;

    std::unique_ptr<double[]> a(new double[m * n]);
    std::unique_ptr<double[]> b(new double[n]);
    std::unique_ptr<double[]> c(new double[m]);

    if (a == NULL || b == NULL || c == NULL)
    {
        printf("Error allocate memory!\n");
        exit(1);
    }

#pragma omp parallel for collapse(2)
    for (size_t i = 0; i < m; i++)
    {
        for (size_t j = 0; j < n; j++)
            a[i * n + j] = i + j;
    }

#pragma omp parallel for
    for (size_t j = 0; j < n; j++)
        b[j] = j;


    double t = cpuSecond();
    matrix_vector_product_omp(a, b, c, m, n);
    t = cpuSecond() - t;

    printf("Elapsed time (parallel): %.6f sec.\n", t);
}

int main(int argc, char *argv[])
{
    size_t M = 20000;
    size_t N = 20000;
    if (argc > 1)
        M = atoi(argv[1]);
    if (argc > 2)
        N = atoi(argv[2]);
    run_serial(M, N);
    run_parallel(M, N);
    return 0;
}
