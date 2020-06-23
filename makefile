CXX=g++
CXXFLAGS=-std=c++11 -O3 -mavx2
#CXXFLAGS=-std=c++11 

matrix.x: matrix.cpp
	$(CXX) $(CXXFLAGS) matrix.cpp -o $@ 

asm: matrix.cpp
	$(CXX) $(CXXFLAGS) matrix.cpp -S -fsave-optimization-record
	cat matrix.s | c++filt > matrix_filt.s

run: matrix.x
	./matrix.x

clean:
	rm matrix.x
