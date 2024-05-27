#include <iostream>
#include <vector>
#include <thread>
#include <memory>
#include <ctime>

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
    for (size_t i = 0; i < m; i++)
    {
        c[i] = 0.0;
        for (size_t j = 0; j < n; j++)
            c[i] += a[i * n + j] * b[j];
    }
}

/*
 * matrix_vector_product_thread: Compute a part of matrix-vector product c[m] = a[m][n] * b[n]
 */
void matrix_vector_product_thread(double *a, double *b, double *c, size_t m, size_t n, size_t lb, size_t ub)
{
    for (size_t i = lb; i <= ub; i++)
    {
        c[i] = 0.0;
        for (size_t j = 0; j < n; j++)
            c[i] += a[i * n + j] * b[j];
    }
}

void matrix_vector_product_parallel(double *a, double *b, double *c, size_t m, size_t n, size_t num_threads)
{
    std::vector<std::thread> threads;
    size_t items_per_thread = m / num_threads;

    for (size_t threadid = 0; threadid < num_threads; ++threadid)
    {
        size_t lb = threadid * items_per_thread;
        size_t ub = (threadid == num_threads - 1) ? (m - 1) : (lb + items_per_thread - 1);
        threads.emplace_back(matrix_vector_product_thread, a, b, c, m, n, lb, ub);
    }

    for (auto &t : threads)
    {
        t.join();
    }
}

void run_serial(size_t n, size_t m)
{
    auto a = std::make_unique<double[]>(m * n);
    auto b = std::make_unique<double[]>(n);
    auto c = std::make_unique<double[]>(m);

    for (size_t i = 0; i < m; i++)
    {
        for (size_t j = 0; j < n; j++)
            a[i * n + j] = i + j;
    }

    for (size_t j = 0; j < n; j++)
        b[j] = j;

    double t = cpuSecond();
    matrix_vector_product(a.get(), b.get(), c.get(), m, n);
    t = cpuSecond() - t;

    std::cout << "Elapsed time (serial): " << t << " sec." << std::endl;
}

void run_parallel(size_t n, size_t m, size_t num_threads)
{
    auto a = std::make_unique<double[]>(m * n);
    auto b = std::make_unique<double[]>(n);
    auto c = std::make_unique<double[]>(m);

    for (size_t i = 0; i < m; i++)
    {
        for (size_t j = 0; j < n; j++)
            a[i * n + j] = i + j;
    }

    for (size_t j = 0; j < n; j++)
        b[j] = j;

    double t = cpuSecond();
    matrix_vector_product_parallel(a.get(), b.get(), c.get(), m, n, num_threads);
    t = cpuSecond() - t;

    std::cout << "Elapsed time (parallel): " << t << " sec." << std::endl;
}

int main(int argc, char *argv[])
{
    size_t M = 1000;
    size_t N = 1000;
    size_t num_threads = 1; 
    if (argc > 1)
        M = atoi(argv[1]);
    if (argc > 2)
        N = atoi(argv[2]);
    if (argc > 3)
        num_threads = atoi(argv[3]);
    run_serial(M, N);
    run_parallel(M, N, num_threads);
    return 0;
}
