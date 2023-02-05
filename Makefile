main:
  g++-12 -c main.cpp -fopenmp -o test.o; g++-12 test.o -o test -fopenmp -lpthread
