CXX=g++ -Wall -pedantic -std=c++17
CXX=g++-7 -Wall -pedantic -std=c++17
INCLUDES=-I/usr/include/python3.6m -I.
INCLUDES=-I/usr/local/Cellar/python3/3.6.2/Frameworks/Python.framework/Versions/3.6/include/python3.6m
LDFLAGS=-L/usr/local/Cellar/python3/3.6.2/Frameworks/Python.framework/Versions/3.6/lib

spectrum_analyzer.so: fft.o file.o wrapper.o spectrum_analyzer.cpp
	$(CXX) spectrum_analyzer.cpp fft.o file.o wrapper.o $(INCLUDES) -o spectrum_analyzer.so -shared $(LDFLAGS) -lpython3.6

fft.o: fft.hpp fft.cpp
	$(CXX) -O2 fft.cpp -c

file.o: file.hpp file.cpp
	$(CXX) -O2 file.cpp -c

wrapper.o: wrapper.hpp wrapper.cpp
	$(CXX) -O2 wrapper.cpp $(INCLUDES) -c