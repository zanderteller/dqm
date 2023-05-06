#ifndef UTILS_H_
#define UTILS_H_

#include "matrix.h"

// calculate eigenvalues and eigenvectors for a square real matrix
void EigValsVecs(Matrix<double>& mat, Matrix<double>& eig_vals, Matrix<double>& eig_vecs);

// get clusters from a matrix
void GetClusters(double* mat, int num_rows, int num_cols, double max_dist, int* cluster_idxs);

// find nearest neighbors and distances for each row in a matrix
void NearestNeighbors(double* mat, int num_rows, int num_cols, int* nn_row_nums, double* nn_dists);

#endif //UTILS_H_
