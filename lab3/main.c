#include <iostream>
#include <vector>
#include <thread>
#include <memory>
#include <chrono>

double cpuSecond()
{
    using namespace std::chrono;
    return duration_cast<duration<double>>(system_clock::now().time_since_epoch()).count();
}

/*
 * matrix_vector_product: Compute matrix-vector product c[m] = a[m][n] * b[n]
 */
void matrix_vector_product(double *a, double *b, double *c, size_t m, size_t n, size_t start, size_t end)
{
    for (size_t i = start; i < end; ++i)
    {
        c[i] = 0.0;
        for (size_t j = 0; j < n; ++j)
            c[i] += a[i * n + j] * b[j];
    }
}

void run_parallel(size_t n, size_t m, size_t num_threads)
{
    std::unique_ptr<double[]> a(new double[m * n]);
    std::unique_ptr<double[]> b(new double[n]);
    std::unique_ptr<double[]> c(new double[m]);

    // Заполнение массивов a и b

    for (size_t i = 0; i < m; ++i)
    {
        for (size_t j = 0; j < n; ++j)
            a[i * n + j] = i + j;
    }

    for (size_t j = 0; j < n; ++j)
        b[j] = j;

    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    size_t chunk_size = m / num_threads;
    size_t remainder = m % num_threads;
    size_t start = 0;
    size_t end = 0;

    double t_start = cpuSecond();

    for (size_t i = 0; i < num_threads; ++i)
    {
        end = start + chunk_size + (i < remainder ? 1 : 0);
        threads.emplace_back(matrix_vector_product, a.get(), b.get(), c.get(), m, n, start, end);
        start = end;
    }

    for (auto &thread : threads)
        thread.join();

    double elapsed_time = cpuSecond() - t_start;
    std::cout << "Elapsed time (parallel): " << elapsed_time << " sec." << std::endl;
}

int main(int argc, char *argv[])
{
    size_t M = 20000;
    size_t N = 20000;
    size_t num_threads = std::thread::hardware_concurrency();
    if (argc > 1)
        M = atoi(argv[1]);
    if (argc > 2)
        N = atoi(argv[2]);
    if (argc > 3)
        num_threads = atoi(argv[3]);

    run_parallel(N, M, num_threads);

    return 0;
}
