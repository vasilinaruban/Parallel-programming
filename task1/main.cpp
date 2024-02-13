#include <iostream>
#include <cmath>
#include <cstdlib>

#ifdef FLOAT_ARRAY
typedef float arrayType ;
#else
typedef double arrayType ;
#endif


template<typename T>
void fill_array(T* array, int size) {
    const T period = 2 * M_PI;
    for (int i = 0; i < size; ++i) {
        T angle = (i * period) / size;
        array[i] = sin(angle);
    }
}

template<typename T>
T calculate_sum(T* array, int size) {
    T sum = 0;
    for (int i = 0; i < size; ++i) {
        sum += array[i];
    }
    return sum;
}

int main() {
    const int size = 10000000;
    auto* array = new arrayType [size];
    fill_array(array, size);
    arrayType sum = calculate_sum(array, size);
    std::cout << sum << std::endl;

    return 0;
}
