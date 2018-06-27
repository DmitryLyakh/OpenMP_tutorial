rm *.mod *.out
gfortran -fopenmp -O3 main.F90 -lgomp
