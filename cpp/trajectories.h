#ifndef TRAJECTORIES_H_
#define TRAJECTORIES_H_
#include <complex>

using std::vector;
using std::complex;

void BuildFramesAuto(Array3D &all_frames, Matrix<double> &basis_rows, Matrix<complex<double>> &exph, vector<Matrix<double> > &xops, Matrix<double> &simdag,
					double sigma, double stopping_threshold);

#endif //TRAJECTORIES_H_
