#include "basis.h"
#include "matrix.h"
#include "operators.h"
#include "utils.h"
#include <vector>
#include <algorithm>


// return the squared Euclidean L2 distance between 2 rows of a matrix
double squared_distance(double* mat, int num_rows, int num_cols, int row_idx_1, int row_idx_2)
{
	double temp;
	double sq_dist = 0;
	int arr_idx_1 = row_idx_1 * num_cols;
	int arr_idx_2 = row_idx_2 * num_cols;
	for (int col_idx = 0; col_idx < num_cols; col_idx++)
	{
		temp = mat[arr_idx_1++] - mat[arr_idx_2++];
		sq_dist += temp * temp;
	}
	return sq_dist;
} // end function squared_distance

/*
find the row in the input matrix that is the biggest outlier (defined as the row with the nearest
neighbor that is farthest away).  we do this the simple, brute-force way, comparing every row to
every other row.  (a supposedly more clever approach, using a flavor of incomplete insertion sort,
turned out to be much slower...)

return the index of the biggest-outlier row.

inputs
* mat: double array holding data for a matrix (in row-major order)
* num_rows: number of rows in mat
* num_cols: number of cols in mat

output
* integer index of the biggest-outlier row
*/
int BiggestOutlier(double* mat, int num_rows, int num_cols)
{
	// find the biggest-outlier row: that is, find the row with the farthest-away nearest neighbor
	// 
	vector<double> nn_sq_dists(num_rows);

	#pragma omp parallel for shared(mat, num_rows, num_cols, nn_sq_dists)
	for (int row_idx1 = 0; row_idx1 < num_rows; row_idx1++)
	{
		// find squared nearest-neighbor distance for the given row (row_idx1)
		double min_sq_dist = -1;
		double sq_dist;
		for (int row_idx2 = 0; row_idx2 < num_rows; row_idx2++)
		{
			if (row_idx1 == row_idx2)
				continue; // row cannot be its own nearest neighbor
			sq_dist = squared_distance(mat, num_rows, num_cols, row_idx1, row_idx2);
			if (sq_dist < min_sq_dist || min_sq_dist < 0)
				min_sq_dist = sq_dist;
		} // end for (each comparison row)
		nn_sq_dists[row_idx1] = min_sq_dist;
	} // end for (each row)

	// now find index for max of squared nearest-neighboor distances
	int max_idx = std::max_element(nn_sq_dists.begin(), nn_sq_dists.end()) - nn_sq_dists.begin();

	return max_idx;
} // end function BiggestOutlier


/*
for a given set of rows and a given desired basis size, choose basis rows from all rows such that the
basis rows are as far apart as possible.

for the first basis row:
* if first_basis_row_num >= 0, use it as the first basis row number.
* if first_basis_row_num < 0, find the biggest outlier (that is, the row whose nearest neighbor is
  farthest away), and use that as the first basis row number.

thereafter, iteratively choose the next non-basis row whose closest distance to any current basis row
is largest.

inputs
* rows: double array holding data for a matrix (in row-major order)
* num_rows: number of rows in the 'rows' matrix
* num_cols: number of columns in the 'rows' matrix
* basis_size: desired number of rows in the basis (must be less than num_rows)
* basis_row_nums: int array holding basis row numbers, which we fill in
* first_basis_row_num: first basis row number (see above)

output
* (none)
*/
void ChooseBasisByDistance(double* rows, int num_rows, int num_cols, int basis_size, int* basis_row_nums, int first_basis_row_num)
{
	assert(basis_size < num_rows);

	// select the first basis row 
	int n_basis = 1;
	if (first_basis_row_num >= 0)
		basis_row_nums[0] = first_basis_row_num;
	else
		basis_row_nums[0] = BiggestOutlier(rows, num_rows, num_cols);

	// set up vector of 'other' rows
	vector<int> other_row_num_vec(num_rows);
	// set up vector of min squared distance from the basis for each 'other' row
	vector<double> other_min_sq_dists(num_rows);
	// initialize the vectors
	for (int idx = 0; idx < num_rows; idx++)
	{
		// this vector will decrease in size as we put things in the basis
		other_row_num_vec[idx] = idx;
		// this vector will NOT decrease in size -- it will be a persistent vector of current
		// min squared distance from all current basis rows, indexed by original row number
		other_min_sq_dists[idx] = -1; // start with a negative sentinel value (which functions as a NaN)
	}
	// remove first basis row from other rows
	other_row_num_vec.erase(other_row_num_vec.begin() + basis_row_nums[0]);

	// add basis rows until we reach desired basis size
	int n_other, new_idx;
	double max_min_sq_dist, sq_dist;
	while (n_basis < basis_size)
	{
		n_other = other_row_num_vec.size();

		// for every 'other' row, find the smallest distance to any current basis row
		#pragma omp parallel for shared(n_basis, n_other, rows, num_rows, num_cols, other_row_num_vec, other_min_sq_dists)
		for (int other_idx = 0; other_idx < n_other; other_idx++)
		{
			// for each 'other' row, we only have to check the distance to the latest basis row, keeping track
			// of the min squared distance as we go
			int other_row_idx = other_row_num_vec[other_idx];
			double new_sq_dist = squared_distance(rows, num_rows, num_cols, basis_row_nums[n_basis - 1], other_row_idx);
			if (new_sq_dist < other_min_sq_dists[other_row_idx] || other_min_sq_dists[other_row_idx] < 0)
				other_min_sq_dists[other_row_idx] = new_sq_dist;
		} // end for (each other row)

		// find current 'other' row that is farthest away from any current basis row
		max_min_sq_dist = -1;
		new_idx = -1;
		for (int other_idx = 0; other_idx < n_other; other_idx++)
		{
			sq_dist = other_min_sq_dists[other_row_num_vec[other_idx]];
			if (sq_dist > max_min_sq_dist)
			{
				max_min_sq_dist = sq_dist;
				new_idx = other_idx;
			}
		} // end for (each 'other' row)

		// add new row to basis and remove it from 'other' rows
		basis_row_nums[n_basis++] = other_row_num_vec[new_idx]; // note: post-increment, not pre-increment
		other_row_num_vec.erase(other_row_num_vec.begin() + new_idx);
	} // end while (haven't reached target basis size yet)
} // end ChooseBasisByDistance

/*
for a given set of basis rows and other rows, build basis overlaps for the other rows.

(note: there's a bit of naming confusion, between BuildOverlaps here and MakeOverlaps in operators.cpp)

inputs
* sigma: value of sigma for the Gaussians around each row/point.
* basis rows: double array holding data for the basis rows
* other_rows: double array holding data for the other rows
* num_basis_rows: number of basis rows
* num_other rows: number of other rows
* num_cols: number of columns (must be the same for basis rows and other rows)
* overlaps: double array holding data fo the overlaps, which we fill in
	
output
* (none)
*/
void BuildOverlaps(double sigma, double* basis_rows, double* other_rows, int num_basis_rows,
											int num_other_rows, int num_cols, double* overlaps)
{
	// set up basis rows and other rows as matrices
	Matrix<double> basis_mat(num_cols, num_basis_rows, basis_rows);
	Matrix<double> other_mat(num_cols, num_other_rows, other_rows);

	// get matrix of pairwise basis-basis overlaps
	Matrix<double> basis_overlaps(MakeOverlaps(basis_mat, sigma));

	// get transpose of 'similarity' matrix
	Matrix<double> sim_t(MakeSim(basis_overlaps).transpose());

	// compute the 'other' basis overlaps
	Matrix<double> basis_other_overlaps(MakeOverlaps(basis_mat, other_mat, sigma));

	// convert the 'other' overlaps to the orthonormal basis of eigenstates
	Matrix<double> ortho_overlaps(sim_t * basis_other_overlaps);

	// calculate L2-norm of total overlap for each 'other' row
	#pragma omp parallel for shared(ortho_overlaps, overlaps)
	for (int other_row_idx = 0; other_row_idx < ortho_overlaps.cols(); other_row_idx++) {
		double overlap = 0;
		double tmp;
		for (int eig_vec_idx = 0; eig_vec_idx < ortho_overlaps.rows(); eig_vec_idx++) {
			tmp = ortho_overlaps.get(other_row_idx, eig_vec_idx);
			overlap += tmp * tmp;
		} // end for (each eigen row)
		overlaps[other_row_idx] = sqrt(overlap);
	} // end for (each other row)
} // end function BuildOverlaps
