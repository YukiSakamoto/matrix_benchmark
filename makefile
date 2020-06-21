CXX=g++
CXXFLAGS=-std=c++11 -O3
#CXXFLAGS=-std=c++11 

matrix.x: matrix.cpp
	$(CXX) $(CXXFLAGS) matrix.cpp -o $@ 

run: matrix.x
	./matrix.x

clean:
	rm matrix.x
