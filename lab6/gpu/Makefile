# Имя исполняемого файла
TARGET = jacobi

# Компилятор и флаги компиляции
CXX = pgc++
CXXFLAGS = -acc -Minfo=all -fast

# Флаги для профилирования (если необходимо)
PROF_FLAGS = -ta=tesla:managed
LDLIBS += -lboost_program_options

# Файлы исходного кода и объектные файлы
SRC = main.cpp
OBJ = $(SRC:.cpp=.o)

# Правила сборки
all: $(TARGET)

$(TARGET): $(OBJ)
  $(CXX) $(LDLIBS) $(CXXFLAGS) $(PROF_FLAGS) -o $@ $^

%.o: %.cpp
  $(CXX) $(CXXFLAGS) -c $< -o $@

# Очистка сгенерированных файлов
clean:
  rm -f $(OBJ) $(TARGET)

# Запуск программы
run: $(TARGET)
  ./$(TARGET) --precision 1e-6 --size 1024 --iter 1000000
