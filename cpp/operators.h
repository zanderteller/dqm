#ifndef OPERATORS_H_
#define OPERATORS_H_

#include "matrix.h"
#include <tuple>
#include <complex>

using std::tuple;
using std::complex;

tuple<Matrix<complex<double>>, vector<Matrix<double>>, Matrix<double>> MakeOperators(Matrix<double> mat, int nhambasis, int npotbasis,
																						double sigma, double step, double mass);

Matrix<double> MakeOverlaps(Matrix<double>& mat, double sigma);

Matrix<double> MakeOverlaps(Matrix<double>& mat1, Matrix<double>& mat2, double sigma);

Matrix<double> MakeSim(Matrix<double> overlaps);

tuple<Matrix<double>, Matrix<double>> AggregatePotentialContributions(Matrix<double>& mat, Matrix<double>& mat_ham_sub, double sigma);

#endif //OPERATORS_H_
