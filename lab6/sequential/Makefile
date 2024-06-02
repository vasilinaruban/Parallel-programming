# Имя компилятора
CXX = pgc++

# Опции компиляции
CXXFLAGS = -acc -Minfo=all -std=c++11

# Опции линковки
LDFLAGS = -lboost_program_options

# Имя исполняемого файла
TARGET = jacobi

# Исходные файлы
SRCS = main.cpp

# Объектные файлы (путем замены расширения .cpp на .o)
OBJS = $(SRCS:.cpp=.o)

# Правило по умолчанию: сборка всех целей
all: $(TARGET)

# Правило для сборки исполняемого файла
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS) $(LDFLAGS)

# Правило для сборки объектных файлов
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Правило для очистки сборки
clean:
	rm -f $(OBJS) $(TARGET)

# Псевдонимы для удобства
.PHONY: all clean
