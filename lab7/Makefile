INFO = -Minfo=all
LIBS = -cudalib=cublas -lboost_program_options
GPU = -acc=gpu
CXX = pgc++

all:gpu
  
gpu: main.cpp
	$(CXX) $(GPU) $(INFO) $(LIBS) -o $@ main.cpp 
clean:all
	rm gpu
