Выбор типа `float`

1-й вариант: `make float`

2-й вариант: `cmake CMakeLists.txt -Dfloat=ON`

Исполняемый файл будет иметь название 'main_float'.

Выбор типа `double`

1-й вариант: `make`

2-й вариант: `cmake CMakeLists.txt -Dfloat=OFF`

Исполняемый файл будет иметь название 'main_double'.

Результаты работы программы:

FLOAT: 0.291951

DOUBLE: 4.89582e-11
