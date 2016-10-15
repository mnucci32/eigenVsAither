/*  An comparison of the Eigen and PETSc libraries for matrix multiplication,
    matrix vector multiplication, and matrix inverses.
    Copyright (C) 2016  Michael Nucci (michael.nucci@gmail.com)

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>. */

// aither includes
#include <matrix.hpp>
#include <genArray.hpp>

// eigen includes
#include <Eigen/Dense>

#include <iostream>
#include <iomanip>
#include <numeric>
#include <chrono>
#include <array>


Eigen::MatrixXd FillEigenMatrix(const int &size, double start,
                                const double &diagFactor);
Eigen::Matrix2d FillEigenMatrix2d(double start, const double &diagFactor);
Eigen::VectorXd FillEigenVector(const int &size, double start);
Eigen::Vector2d FillEigenVector2d(double start);

squareMatrix FillAitherMatrix(const int &size, double start,
                              const double &diagFactor);
genArray FillAitherVector(const int &size, double start);
genArray FillAitherVector2d(double start);


int main(int argc, char *argv[]) {
  // determine if eigen is vectorized
#ifdef EIGEN_VECTORIZE
  std::cout << "Eigen is vectorized" << std::endl;
#endif

  // put data into eigen matrices and vectors
  // eigen contains special fixed sized matrices and vectors which allows
  // allocation on stack -- will test static and dynamic allocation versions
  auto eFlowMat = FillEigenMatrix(5, -6.0, 10.0);
  auto eTurbMat = FillEigenMatrix2d(1.0, 10.0);
  auto eXTurbMat = FillEigenMatrix(2, 1.0, 10.0);
  auto eFlowVec = FillEigenVector(5, -9.0);
  auto eTurbVec = FillEigenVector2d(3.0);
  auto eXTurbVec = FillEigenVector(2, 3.0);
  std::cout << "***** Populating Eigen Data *****\n" << std::endl;
  std::cout << "Eigen flow matrix:\n" << eFlowMat << std::endl;
  std::cout << "\nEigen turbulence matrix (static):\n" << eTurbMat << std::endl;
  std::cout << "\nEigen flow vector:\n" << eFlowVec << std::endl;
  std::cout << "\nEigen turbulence vector (static):\n" << eTurbVec << std::endl;
  std::cout << "\nEigen turbulence matrix (dynamic):\n" << eXTurbMat
            << std::endl;
  std::cout << "\nEigen turbulence vector (dynamic):\n" << eXTurbVec
            << std::endl;
  std::cout << "\n********************\n" << std::endl;

  // put data into aither matrices and vectors
  // matrices in aither are variable size and dynamically allocated
  // vectors in aither are fixed size (max equations) and statically allocated
  auto aFlowMat = FillAitherMatrix(5, -6.0, 10.0);
  auto aTurbMat = FillAitherMatrix(2, 1.0, 10.0);
  auto aFlowVec = FillAitherVector(5, -9.0);
  auto aTurbVec = FillAitherVector2d(3.0);
  std::cout << "***** Populating Aither Data *****\n" << std::endl;
  std::cout << "Aither flow matrix:\n" << aFlowMat << std::endl;
  std::cout << "\nAither turbulence matrix:\n" << aTurbMat << std::endl;
  std::cout << "\nAither flow vector:\n" << aFlowVec << std::endl;
  std::cout << "\nAither turbulence vector:\n" << aTurbVec << std::endl;
  std::cout << "\n********************\n" << std::endl;

  constexpr auto samples = 10000000;

  // array to hold eigen times
  std::array<double, 5> eTimeFlow;
  std::array<double, 5> eTimeTurb;
  std::array<double, 5> eXTimeTurb;
  // array to hold aither times
  std::array<double, 5> aTimeFlow;
  std::array<double, 5> aTimeTurb;

  // test matrix matrix multiplication ---------------------------------------
  // eigen
  Eigen::MatrixXd eFlowProd;
  Eigen::Matrix2d eTurbProd;
  Eigen::MatrixXd eXTurbProd;

  auto start = std::chrono::high_resolution_clock::now();
  for (auto ii = 0; ii < samples; ++ii) {
    eFlowProd = eFlowMat * eFlowMat;
  }
  auto stop = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = stop - start;
  std::cout << "Eigen flow matrix matrix multiplication: " << duration.count()
            << " seconds" << std::endl;
  std::cout << eFlowProd << "\n" << std::endl;
  eTimeFlow[0] = duration.count();

  start = std::chrono::high_resolution_clock::now();
  for (auto ii = 0; ii < samples; ++ii) {
    eTurbProd = eTurbMat * eTurbMat;
  }
  stop = std::chrono::high_resolution_clock::now();
  duration = stop - start;
  std::cout << "Eigen (static) turbulence matrix matrix multiplication: "
            << duration.count() << " seconds" << std::endl;
  std::cout << eTurbProd << "\n" << std::endl;
  eTimeTurb[0] = duration.count();

  start = std::chrono::high_resolution_clock::now();
  for (auto ii = 0; ii < samples; ++ii) {
    eXTurbProd = eXTurbMat * eXTurbMat;
  }
  stop = std::chrono::high_resolution_clock::now();
  duration = stop - start;
  std::cout << "Eigen (dynamic) turbulence matrix matrix multiplication: "
            << duration.count() << " seconds" << std::endl;
  std::cout << eXTurbProd << "\n" << std::endl;
  eXTimeTurb[0] = duration.count();

  // aither
  squareMatrix aFlowProd;
  squareMatrix aTurbProd;
  start = std::chrono::high_resolution_clock::now();
  for (auto ii = 0; ii < samples; ++ii) {
    aFlowProd = aFlowMat.MatMult(aFlowMat);
  }
  stop = std::chrono::high_resolution_clock::now();
  duration = stop - start;
  std::cout << "Aither flow matrix matrix multiplication: " << duration.count()
            << " seconds" << std::endl;
  std::cout << aFlowProd << "\n" << std::endl;
  aTimeFlow[0] = duration.count();

  start = std::chrono::high_resolution_clock::now();
  for (auto ii = 0; ii < samples; ++ii) {
    aTurbProd = aTurbMat.MatMult(aTurbMat);
  }
  stop = std::chrono::high_resolution_clock::now();
  duration = stop - start;
  std::cout << "Aither turbulence matrix matrix multiplication: "
            << duration.count() << " seconds" << std::endl;
  std::cout << aTurbProd << "\n" << std::endl;
  aTimeTurb[0] = duration.count();

  // test matrix vector multiplication --------------------------------------
  // eigen
  Eigen::VectorXd eFlowProdV;
  Eigen::Vector2d eTurbProdV;
  Eigen::VectorXd eXTurbProdV;

  start = std::chrono::high_resolution_clock::now();
  for (auto ii = 0; ii < samples; ++ii) {
    eFlowProdV = eFlowMat * eFlowVec;
  }
  stop = std::chrono::high_resolution_clock::now();
  duration = stop - start;
  std::cout << "Eigen flow matrix vector multiplication: " << duration.count()
            << " seconds" << std::endl;
  std::cout << eFlowProdV << "\n" << std::endl;
  eTimeFlow[1] = duration.count();

  start = std::chrono::high_resolution_clock::now();
  for (auto ii = 0; ii < samples; ++ii) {
    eTurbProdV = eTurbMat * eTurbVec;
  }
  stop = std::chrono::high_resolution_clock::now();
  duration = stop - start;
  std::cout << "Eigen (static) turbulence matrix vector multiplication: "
            << duration.count() << " seconds" << std::endl;
  std::cout << eTurbProdV << "\n" << std::endl;
  eTimeTurb[1] = duration.count();

  start = std::chrono::high_resolution_clock::now();
  for (auto ii = 0; ii < samples; ++ii) {
    eXTurbProdV = eXTurbMat * eXTurbVec;
  }
  stop = std::chrono::high_resolution_clock::now();
  duration = stop - start;
  std::cout << "Eigen (dynamic) turbulence matrix vector multiplication: "
            << duration.count() << " seconds" << std::endl;
  std::cout << eXTurbProdV << "\n" << std::endl;
  eXTimeTurb[1] = duration.count();

  // aither
  genArray aFlowProdV;
  genArray aTurbProdV;

  start = std::chrono::high_resolution_clock::now();
  for (auto ii = 0; ii < samples; ++ii) {
    aFlowProdV = aFlowMat.ArrayMult(aFlowVec);
  }
  stop = std::chrono::high_resolution_clock::now();
  duration = stop - start;
  std::cout << "Eigen flow matrix vector multiplication: " << duration.count()
            << " seconds" << std::endl;
  std::cout << aFlowProdV << "\n" << std::endl;
  aTimeFlow[1] = duration.count();

  start = std::chrono::high_resolution_clock::now();
  for (auto ii = 0; ii < samples; ++ii) {
    aTurbProdV = aTurbMat.ArrayMult(aTurbVec, 5);
  }
  stop = std::chrono::high_resolution_clock::now();
  duration = stop - start;
  std::cout << "Eigen turbulence matrix vector multiplication: "
            << duration.count() << " seconds" << std::endl;
  std::cout << aTurbProdV << "\n" << std::endl;
  aTimeTurb[1] = duration.count();

  // test matrix multiplication with scalar and addition --------------------
  // eigen
  Eigen::MatrixXd eFlowAdd;
  Eigen::Matrix2d eTurbAdd;
  Eigen::MatrixXd eXTurbAdd;

  start = std::chrono::high_resolution_clock::now();
  for (auto ii = 0; ii < samples; ++ii) {
    eFlowAdd = 1.5 * eFlowMat + eFlowMat;
  }
  stop = std::chrono::high_resolution_clock::now();
  duration = stop - start;
  std::cout << "Eigen flow matrix matrix addition: " << duration.count()
            << " seconds" << std::endl;
  std::cout << eFlowAdd << "\n" << std::endl;
  eTimeFlow[2] = duration.count();

  start = std::chrono::high_resolution_clock::now();
  for (auto ii = 0; ii < samples; ++ii) {
    eTurbAdd = 1.5 * eTurbMat + eTurbMat;
  }
  stop = std::chrono::high_resolution_clock::now();
  duration = stop - start;
  std::cout << "Eigen (static) turbulence matrix matrix addition: "
            << duration.count() << " seconds" << std::endl;
  std::cout << eTurbAdd << "\n" << std::endl;
  eTimeTurb[2] = duration.count();

  start = std::chrono::high_resolution_clock::now();
  for (auto ii = 0; ii < samples; ++ii) {
    eXTurbAdd = 1.5 * eXTurbMat + eXTurbMat;
  }
  stop = std::chrono::high_resolution_clock::now();
  duration = stop - start;
  std::cout << "Eigen (dynamic) turbulence matrix matrix addition: "
            << duration.count() << " seconds" << std::endl;
  std::cout << eXTurbAdd << "\n" << std::endl;
  eXTimeTurb[2] = duration.count();

  // aither
  squareMatrix aFlowAdd;
  squareMatrix aTurbAdd;

  start = std::chrono::high_resolution_clock::now();
  for (auto ii = 0; ii < samples; ++ii) {
    aFlowAdd = 1.5 * aFlowMat + aFlowMat;
  }
  stop = std::chrono::high_resolution_clock::now();
  duration = stop - start;
  std::cout << "Aither flow matrix matrix addition: " << duration.count()
            << " seconds" << std::endl;
  std::cout << aFlowAdd << "\n" << std::endl;
  aTimeFlow[2] = duration.count();

  start = std::chrono::high_resolution_clock::now();
  for (auto ii = 0; ii < samples; ++ii) {
    aTurbAdd = 1.5 * aTurbMat + aTurbMat;
  }
  stop = std::chrono::high_resolution_clock::now();
  duration = stop - start;
  std::cout << "Aither turbulence matrix matrix addition: "
            << duration.count() << " seconds" << std::endl;
  std::cout << aTurbAdd << "\n" << std::endl;
  aTimeTurb[2] = duration.count();

  // test vector multiplication with scalar and addition --------------------
  // eigen
  Eigen::VectorXd eFlowAddV;
  Eigen::Vector2d eTurbAddV;
  Eigen::VectorXd eXTurbAddV;

  start = std::chrono::high_resolution_clock::now();
  for (auto ii = 0; ii < samples; ++ii) {
    eFlowAddV = 1.5 * eFlowVec + eFlowVec;
  }
  stop = std::chrono::high_resolution_clock::now();
  duration = stop - start;
  std::cout << "Eigen flow vector vector addition: " << duration.count()
            << " seconds" << std::endl;
  std::cout << eFlowAddV << "\n" << std::endl;
  eTimeFlow[3] = duration.count();

  start = std::chrono::high_resolution_clock::now();
  for (auto ii = 0; ii < samples; ++ii) {
    eTurbAddV = 1.5 * eTurbVec + eTurbVec;
  }
  stop = std::chrono::high_resolution_clock::now();
  duration = stop - start;
  std::cout << "Eigen (static) turbulence vector vector addition: "
            << duration.count() << " seconds" << std::endl;
  std::cout << eTurbAddV << "\n" << std::endl;
  eTimeTurb[3] = duration.count();

  start = std::chrono::high_resolution_clock::now();
  for (auto ii = 0; ii < samples; ++ii) {
    eXTurbAddV = 1.5 * eXTurbVec + eXTurbVec;
  }
  stop = std::chrono::high_resolution_clock::now();
  duration = stop - start;
  std::cout << "Eigen (dynamic) turbulence vector vector addition: "
            << duration.count() << " seconds" << std::endl;
  std::cout << eXTurbAddV << "\n" << std::endl;
  eXTimeTurb[3] = duration.count();

  // aither
  genArray aFlowAddV;

  start = std::chrono::high_resolution_clock::now();
  for (auto ii = 0; ii < samples; ++ii) {
    aFlowAddV = 1.5 * aFlowVec + aFlowVec;
  }
  stop = std::chrono::high_resolution_clock::now();
  duration = stop - start;
  std::cout << "Aither flow vector vector addition: " << duration.count()
            << " seconds" << std::endl;
  std::cout << aFlowAddV << "\n" << std::endl;
  aTimeFlow[3] = duration.count();

  // in aither vectors are of size 7, so no comparison for turbulence


  // test matrix inverse ---------------------------------------------------
  // eigen
  Eigen::MatrixXd eFlowInv;
  Eigen::Matrix2d eTurbInv;
  Eigen::MatrixXd eXTurbInv;

  start = std::chrono::high_resolution_clock::now();
  for (auto ii = 0; ii < samples; ++ii) {
    eFlowInv = eFlowMat.inverse();
  }
  stop = std::chrono::high_resolution_clock::now();
  duration = stop - start;
  std::cout << "Eigen flow matrix inversion: " << duration.count()
            << " seconds" << std::endl;
  std::cout << eFlowInv << "\n" << std::endl;
  eTimeFlow[4] = duration.count();

  start = std::chrono::high_resolution_clock::now();
  for (auto ii = 0; ii < samples; ++ii) {
    eTurbInv = eTurbMat.inverse();
  }
  stop = std::chrono::high_resolution_clock::now();
  duration = stop - start;
  std::cout << "Eigen (static) turbulence matrix inversion: "
            << duration.count() << " seconds" << std::endl;
  std::cout << eTurbInv << "\n" << std::endl;
  eTimeTurb[4] = duration.count();

  start = std::chrono::high_resolution_clock::now();
  for (auto ii = 0; ii < samples; ++ii) {
    eXTurbInv = eXTurbMat.inverse();
  }
  stop = std::chrono::high_resolution_clock::now();
  duration = stop - start;
  std::cout << "Eigen (dynamic) turbulence matrix inversion: "
            << duration.count() << " seconds" << std::endl;
  std::cout << eXTurbInv << "\n" << std::endl;
  eXTimeTurb[4] = duration.count();

  // aither
  squareMatrix aFlowInv;
  squareMatrix aTurbInv;

  start = std::chrono::high_resolution_clock::now();
  for (auto ii = 0; ii < samples; ++ii) {
    aFlowInv = aFlowMat;
    aFlowInv.Inverse();
  }
  stop = std::chrono::high_resolution_clock::now();
  duration = stop - start;
  std::cout << "Aither flow matrix inversion: " << duration.count()
            << " seconds" << std::endl;
  std::cout << aFlowInv << "\n" << std::endl;
  aTimeFlow[4] = duration.count();

  start = std::chrono::high_resolution_clock::now();
  for (auto ii = 0; ii < samples; ++ii) {
    aTurbInv = aTurbMat;
    aTurbInv.Inverse();
  }
  stop = std::chrono::high_resolution_clock::now();
  duration = stop - start;
  std::cout << "Aither turbulence matrix inversion: "
            << duration.count() << " seconds" << std::endl;
  std::cout << aTurbInv << "\n" << std::endl;
  aTimeTurb[4] = duration.count();

  // print timing summary --------------------------------------------------
  std::cout << "Timing Summary" << std::endl;
  std::cout << std::scientific << std::setprecision(5);
  std::cout << std::setw(25) << "Matrix Type";
  std::cout << std::setw(15) << "MatMult" << std::setw(15) << "MatVec"
            << std::setw(15) << "MatAdd" << std::setw(15) << "VecAdd"
            << std::setw(15) << "Inverse" << std::endl;
  std::cout << std::setw(25) << "Eigen 5x5 Dynamic";
  std::cout << std::setw(15) << eTimeFlow[0] << std::setw(15) << eTimeFlow[1]
            << std::setw(15) << eTimeFlow[2] << std::setw(15) << eTimeFlow[3]
            << std::setw(15) << eTimeFlow[4] << std::endl;
  std::cout << std::setw(25) << "Eigen 2x2 Static";
  std::cout << std::setw(15) << eTimeTurb[0] << std::setw(15) << eTimeTurb[1]
            << std::setw(15) << eTimeTurb[2] << std::setw(15) << eTimeTurb[3]
            << std::setw(15) << eTimeTurb[4] << std::endl;
  std::cout << std::setw(25) << "Eigen 2x2 Dynamic";
  std::cout << std::setw(15) << eXTimeTurb[0] << std::setw(15) << eXTimeTurb[1]
            << std::setw(15) << eXTimeTurb[2] << std::setw(15) << eXTimeTurb[3]
            << std::setw(15) << eXTimeTurb[4] << std::endl;

  std::cout << std::setw(25) << "Aither 5x5 Dynamic";
  std::cout << std::setw(15) << aTimeFlow[0] << std::setw(15) << aTimeFlow[1]
            << std::setw(15) << aTimeFlow[2] << std::setw(15) << aTimeFlow[3]
            << std::setw(15) << aTimeFlow[4] << std::endl;
  std::cout << std::setw(25) << "Aither 2x2 Dynamic";
  std::cout << std::setw(15) << aTimeTurb[0] << std::setw(15) << aTimeTurb[1]
            << std::setw(15) << aTimeTurb[2] << std::setw(15) << aTimeTurb[3]
            << std::setw(15) << aTimeTurb[4] << std::endl;
  return 0;
}

// -------------------------------------------------------------------------
// function definitions ----------------------------------------------------
Eigen::MatrixXd FillEigenMatrix(const int &size, double start,
                                const double &diagFactor) {
  Eigen::MatrixXd matrix(size, size);
  for (auto rr = 0; rr < size; ++rr) {
    for (auto cc = 0; cc < size; ++cc) {
      auto fac = (rr == cc) ? diagFactor : 1.0;
      matrix(rr, cc) = ++start * fac;
    }
  }
  return matrix;
}

Eigen::Matrix2d FillEigenMatrix2d(double start, const double &diagFactor) {
  Eigen::Matrix2d matrix;
  for (auto rr = 0; rr < 2; ++rr) {
    for (auto cc = 0; cc < 2; ++cc) {
      auto fac = (rr == cc) ? diagFactor : 1.0;
      matrix(rr, cc) = ++start * fac;
    }
  }
  return matrix;
}

Eigen::VectorXd FillEigenVector(const int &size, double start) {
  Eigen::VectorXd vector(size);
  for (auto rr = 0; rr < size; ++rr) {
    vector(rr) = ++start;
  }
  return vector;
}

Eigen::Vector2d FillEigenVector2d(double start) {
  Eigen::Vector2d vector;
  for (auto rr = 0; rr < 2; ++rr) {
    vector(rr) = ++start;
  }
  return vector;
}

squareMatrix FillAitherMatrix(const int &size, double start,
                              const double &diagFactor) {
  squareMatrix matrix(size);
  for (auto rr = 0; rr < size; ++rr) {
    for (auto cc = 0; cc < size; ++cc) {
      auto fac = (rr == cc) ? diagFactor : 1.0;
      matrix(rr, cc) = ++start * fac;
    }
  }
  return matrix;
}


genArray FillAitherVector(const int &size, double start) {
  genArray vector(0.0);
  for (auto rr = 0; rr < size; ++rr) {
    vector[rr] = ++start;
  }
  return vector;
}

genArray FillAitherVector2d(double start) {
  genArray vector(0.0);
  for (auto rr = 5; rr < 7; ++rr) {
    vector[rr] = ++start;
  }
  return vector;
}

