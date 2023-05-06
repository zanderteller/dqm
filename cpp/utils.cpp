/********************

General utility functions.

********************/

#include <assert.h>
#include "utils.h"
#include "matrix.h"
#include <cassert>
#include <cmath>
#include <Eigen/Dense>

/********************

Eigenvalue/eigenvector functions.

********************/

/*
calculate eigenvalues and eigenvectors for a square real matrix

inputs
* mat: the square real matrix
* eig_vals: a vector, which will be populated with the eigenvalues, in ascending order
* eig_vecs: a square matrix, will be populated with the associated eigenvectors as columns of the matrix
*/
void EigValsVecs(Matrix<double>& mat, Matrix<double>& eig_vals, Matrix<double>& eig_vecs)
{
	// verify correct size of matrices
	int n = mat.rows();
	assert(n == mat.cols() && n == eig_vals.rows() && 1 == eig_vals.cols() && n == eig_vecs.rows() && n == eig_vecs.cols());

	// create Eigen matrices, without allocating new memory (using Map)
	Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> mat_e(mat.raw(), mat.rows(), mat.cols());
	// note: Eigen requires that column vectors be stored in column-major order
	Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1, Eigen::ColMajor>> eig_vals_e(eig_vals.raw(), eig_vals.rows());
	Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eig_vecs_e(eig_vecs.raw(), eig_vecs.rows(), eig_vecs.cols());

	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(mat_e);
	eig_vals_e = es.eigenvalues();
	eig_vecs_e = es.eigenvectors();
} // end function EigValsVecs

/********************

Clustering functions.

********************/

double SquaredDistance(double* mat, int num_rows, int num_cols, int row_idx_1, int row_idx_2)
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
} // end function SquaredDistance

/*
return the L2 distance and the row number for each row's nearest-neighbor

inputs
* mat: double array containing an <n x d> matrix
* num_rows: n
* num_cols: d
* nn_row_nums: allocated int array, which we populate with the row number of the nearest neighbor for each row.
* nn_dists: allocated double array, which we populate with the nearest-neighbor L2 distance for each row.

output
* none
*/
void NearestNeighbors(double* mat, int num_rows, int num_cols, int* nn_row_nums, double* nn_dists)
{
	# pragma omp parallel for shared(mat, num_rows, num_cols, nn_row_nums, nn_dists)
	for (int row_num = 0; row_num < num_rows; row_num++)
	{
		int nn_row_num = -1;
		double min_sq_dist = -1;
		double sq_dist;
		for (int row_num_2 = 0; row_num_2 < num_rows; row_num_2++)
		{
			if (row_num_2 == row_num)
				continue; // row cannot be its own nearest neighbor
			sq_dist = SquaredDistance(mat, num_rows, num_cols, row_num, row_num_2);
			if (sq_dist < min_sq_dist || min_sq_dist < 0)
			{
				min_sq_dist = sq_dist;
				nn_row_num = row_num_2;
			} // end if found new min squared distance
		} // end for each comparison row
		nn_row_nums[row_num] = nn_row_num;
		nn_dists[row_num] = sqrt(min_sq_dist);
	} // end for each row
} // end function NearestNeighbors

// find unclaimed rows within max_dist of seed row and mark them with the sentinel value
int GrowClusterFromSeed(double* mat, int num_rows, int num_cols, double max_dist, int* cluster_idxs, int start_row_idx, int seed_row_idx, int sentinel_value)
{
	// sentinel values may already exist in cluster_idxs -- use a temporary sentinel value to track how many we added here
	int temp_sentinel_value = 2 * num_rows;

	double sq_max_dist = max_dist * max_dist;

	#pragma omp parallel for shared(mat, num_rows, num_cols, sq_max_dist, cluster_idxs, start_row_idx, seed_row_idx, sentinel_value)
	for (int row_idx = start_row_idx; row_idx < num_rows; row_idx++)
	{
		if (cluster_idxs[row_idx] == -1 && row_idx != seed_row_idx)
		{
			// row is unclaimed -- see if we can claim it
			double sq_dist = SquaredDistance(mat, num_rows, num_cols, seed_row_idx, row_idx);
			if (sq_dist <= sq_max_dist)
				cluster_idxs[row_idx] = temp_sentinel_value;
		} // end if row is unclaimed
	} // end for each row after seed row

	// count up how many rows were flagged, and mark each one with the real sentinel value
	int num_added = 0;
	for (int row_idx = start_row_idx; row_idx < num_rows; row_idx++)
		if (cluster_idxs[row_idx] == temp_sentinel_value)
		{
			cluster_idxs[row_idx] = sentinel_value;
			num_added++;
		}

	return num_added;
} // end function GrowClusterFromSeed

// find unclaimed rows within max_dist of any sentinel-value rows after seed row and mark them with the next lower sentinel value
int GrowClusterFromSentinelValue(double* mat, int num_rows, int num_cols, int num_candidates, double max_dist, int* cluster_idxs, int start_row_idx, int sentinel_value)
{
	int total_added = 0;
	int num_added;

	// for each row marked with sentinel_value, grow from that row as seed row and mark new rows with sentinel_value - 1
	for (int row_idx = start_row_idx; row_idx < num_rows; row_idx++)
	{
		if (num_candidates == 0)
			break;
		if (cluster_idxs[row_idx] == sentinel_value)
		{
			num_added = GrowClusterFromSeed(mat, num_rows, num_cols, max_dist, cluster_idxs, start_row_idx, row_idx, sentinel_value - 1);
			total_added += num_added;
			num_candidates -= num_added;
		}
	}

	return total_added;
} // end function GrowClusterFromSentinelValue

/*
get a single cluster from a matrix, starting from the given start row.  we here define a cluster as a group of rows
such that every row is within max_dist Euclidean distance of at least one other row in the cluster.  the largest distance
between 2 rows in a cluster may be much larger than max_dist.  ('cluster' is defined here in a way that will include
extended structures.)

The output in cluster_idxs represents each cluster as a linked list, as follows: starting from row 0, each row's entry
in cluster_idxs contains the row idx of the next row in this row's cluster.  the last row (and possibly first row) in any
cluster will have num_rows as its entry in cluster_idxs.  any row with a -1 in cluster_idxs is still unclaimed.

inputs
* mat: double array containing the <num_rows x num_cols> matrix entries
* num_rows: number of rows in mat
* num_cols: number of columns inmat
* num_candidates: number of unclaimed rows, including start row, none of which will be before start row
* max_dist: maximum Euclidean for membership in a cluster.
* cluster_idxs: allocated <num_rows> int array, which we fill in.
* start_row_idx: index for the row to use as the seed for cluster.

output
* integer number of rows in the cluster
*/
int GetCluster(double* mat, int num_rows, int num_cols, int num_candidates, double max_dist, int* cluster_idxs, int start_row_idx)
{
	/*
	algo notes

	the plan:
	* in cluster_idxs, -1 is the sentinel value telling us that a row is unclaimed (not yet part of a cluster)
	* mark start row with a sentinel value of -2
	* call GrowClusterFromSeed to mark all unclaimed rows within max_dist of the start row with a -3 sentinel value
	* call GrowClusterFromSentinelValue with a sentinel value of -2 -- this will mark all unclaimed rows within max_dist
	  of the -2 rows with a -3.  then call GrowClusterFromSentinelValue again with a value of -3, etc.  Continue as long as
	  new rows were added with the latest sentinel value.
	* walk forward from start row, building linked list for the cluster using all cluster index values < -1.
	* mark the last row in the cluster with num_rows as its entry in cluster_idxs

	*/

	// mark start row with sentinel value of -2
	int sentinel_value = -2;
	cluster_idxs[start_row_idx] = sentinel_value;
	int num_in_cluster = 1;
	num_candidates--;

	int num_added;
	bool done = false;
	while (!done)
	{
		// GrowClusterFromSentinelValue labels newly claimed rows with (sentinel-value - 1)
		num_added = GrowClusterFromSentinelValue(mat, num_rows, num_cols, num_candidates, max_dist, cluster_idxs, start_row_idx, sentinel_value);
		done = num_added == 0;
		num_in_cluster += num_added;
		num_candidates -= num_added;
		sentinel_value--;
	} // end while not done

	int last_row_idx = start_row_idx;
	int num_left = num_in_cluster;
	if (num_left > 1)
	{
		// walk forward and build the linked list for this cluster -- any row with a sentinel value < -1 is part of the cluster
		// (note: we can ignore rows before start row, since they will all have been claimed already)
		for (int row_idx = start_row_idx + 1; row_idx < num_rows; row_idx++)
		{
			if (cluster_idxs[row_idx] < -1)
			{
				// this row was claimed for this cluster
				cluster_idxs[last_row_idx] = row_idx; // last row points to this row
				last_row_idx = row_idx; // this row is now last row
				num_left--;
				if (num_left == 1)
					break; // last row is dealt with outside the loop
			} // end if adding row to the cluster
		} // end for each row after start row
	} // end if any rows added after the start row
	cluster_idxs[last_row_idx] = num_rows; // mark the end of the linked list with the out-of-range value num_rows

	return num_in_cluster;
} // end function GetCluster

/*
get all clusters from a matrix.  we here define a cluster as a group of rows such that every row is within max_dist
Euclidean distance of at least one other row in the cluster.  the largest distance between 2 rows in a cluster may be
much larger than max_dist.  ('cluster' is defined here in a way that will include extended structures.)

The output in cluster_idxs represents each cluster as a linked list, as follows: starting from row 0, each row's entry
in cluster_idxs contains the row idx of the next row in this row's cluster.  the last row (and possibly first row) in any
cluster will have num_rows as its entry in cluster_idxs.  any row with a -1 in cluster_idxs is still unclaimed.

inputs
* mat: double array containing the <num_rows x num_cols> matrix entries
* num_rows: number of rows in mat
* num_cols: number of columns inmat
* max_dist: maximum Euclidean for membership in a cluster.
* cluster_idxs: allocated <num_rows> int array, which we fill in.

output
* none
*/
void GetClusters(double* mat, int num_rows, int num_cols, double max_dist, int* cluster_idxs)
{
	// initialize all cluster indices to -1
	for (int i = 0; i < num_rows; i++)
		cluster_idxs[i] = -1;

	int num_candidates = num_rows;
	int num_in_cluster;

	for (int start_row_idx = 0; start_row_idx < num_rows; start_row_idx++)
	{
		if (cluster_idxs[start_row_idx] == -1)
		{
			// this row is unclaimed -- build a cluster with this row as the seed row
			// note: GetCluster will never look before start_row_idx -- any earlier row will already be in a cluster
			num_in_cluster = GetCluster(mat, num_rows, num_cols, num_candidates, max_dist, cluster_idxs, start_row_idx);
			num_candidates -= num_in_cluster;
		}
	}
} // end function GetClusters
