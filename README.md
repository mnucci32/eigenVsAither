# eigenVsAither
An evaluation of the Eigen linear algebra library for use in the aither CFD solver

## Motivation
CFD solvers such as [aither](https://github.com/mnucci32/aither) make frequent use of matrix and vector operations, so any speed up to these operations can greatly improve performance. Currently aither does not leverage any third party linear algebra library. The [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page) library is a template library for linear algebra written in C++. It is widely used and has the benefits of being header only so there is no need to link to a library. It can also be distributed with project source code which can alleviate the problems associated with trying to find different libraries on different computer systems. The purpose of this program is to test the performance of the Eigen library versus the code already in aither for some common matrix and vector operations used in a CFD code.

## Tests
This code performs five tests on matrices and vectors of two different sizes. The matrix sizes are 5x5 and 2x2, the vector sizes are 5 and 2. These sizes were chosen because in an implicit CFD flux jacobian matrices must be calculated and stored, and for the 3D Navier-Stokes equations, these matrices are 5x5. For a standard two equation turbulence model used in the Reynolds-Averaged Navier-Stokes equations, the flux jacobian matrices are 2x2. The five tests performed are matrix-matrix multiplication, matrix-vector multiplication, matrix multiplication by a scalar and addition with another matrix, vector multiplication by a scalar and addition with another vector, and matrix inverse.

## Compilation
This code uses cmake and assumes that you have aither and Eigen on your system. To compile use the process below

```(bash)
cmake -DAITHER_DIR=/path/to/aither -DEIGEN_DIR=/path/to/eigen -DCMAKE_BUILD_TYPE=release /path/to/source
make
```
## Results
Results show that for these five tests, the aither code outperforms the Eigen code unless the Eigen code is using its statically allocated matrices and vectors of predetermined size. This is surprising, but according to the Eigen [benchmarks](http://eigen.tuxfamily.org/index.php?title=Benchmark), Eigen's best performance is for larger sized matrices. Using the statically allocated matrices and vectors is not an option for eigen because these matrix sizes are determined at run time. Therefore Eigen will not be incorporated into aither.

