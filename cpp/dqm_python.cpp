#define  _CRT_SECURE_NO_WARNINGS

#include <complex>
#include <vector>

#include "utils.h"
#include "basis.h"
#include "matrix.h"
#include "array3d.h"
#include "operators.h"
#include "trajectories.h"

using std::complex;
using std::tuple;
using std::get;

#ifdef _WIN32
#include <Windows.h>
BOOL APIENTRY DllMain(HMODULE hModule, DWORD  ul_reason_for_call, LPVOID lpReserved)
{
	return TRUE;
}
#define DLLEXPORT __declspec(dllexport)
#else
#define __declspec(A)
#define DLLEXPORT
#endif

extern "C"
{
	DLLEXPORT void GetClustersC(double* mat, int num_rows, int num_cols, double max_dist, int* cluster_idxs)
	{
		GetClusters(mat, num_rows, num_cols, max_dist, cluster_idxs);
	} // end function GetClustersC

	DLLEXPORT void NearestNeighborsC(double* mat, int num_rows, int num_cols, int* nn_row_nums, double* nn_dists)
	{
		NearestNeighbors(mat, num_rows, num_cols, nn_row_nums, nn_dists);
	} // end function NearestNeighborsC

	DLLEXPORT void ChooseBasisByDistanceC(double* rows, int num_rows, int num_cols, int basis_size, int* basis_row_nums, int first_basis_row_num)
	{
		ChooseBasisByDistance(rows, num_rows, num_cols, basis_size, basis_row_nums, first_basis_row_num);
	} // end function ChhoseBAsisByDistanceC

	DLLEXPORT void BuildOverlapsC(double sigma, double* basis_rows, double* other_rows, int num_basis_rows, int num_other_rows, int num_cols, double* overlaps)
	{
		BuildOverlaps(sigma, basis_rows, other_rows, num_basis_rows, num_other_rows, num_cols, overlaps);
	} // end function BuildOverlapsC	

	/*
	create and return DQM operators
	
	inputs
	* mat: double array containing the rows in frame 1
	* num_rows: number of rows in mat
	* num_cols: number of columns (i.e., dimensions) in mat
	* nhambasis: use this many rows (starting from first row) as the basis
	* npotbasis: use this many rows (starting from first row) to build the potential
	* sigma: value of sigma
	* step: value of time step
	* mass: value of mass
	* sim_t: output, which we fill in -- <nhambasis x nhambasis> double matrix that is the transpose
	  of the 'similarity' matrix, used to to convert from representation in the 'raw' basis of basis
	  rows to representation in the orthonormal basis of eigenstates
	* xops: output, which we fill in -- <nhambasis x nhambasis x num_cols> double array that is the
	  position-expectation operators (one for each dimension)
	* eth: output, which we fill in -- <nhambasis x nhambasis> complex matrix that is the Hamiltonian
	  time-evolution exponentiation operator

	output
	* n_eigstates: integer number of basis eigenstates, which may be smaller than nhambasis.
	  NOTE: if n_eigstates is smaller than nhambasis, calling code should subselect the 3 output arrays
	  accordingly.  whenever the 'real' size of a dimension is n_eigstates, data is written to the first
	  n_eigstates rows or columns in that dimension.  here are the 'real' dimensions of each output array:
		- simt: <n_eigstates x nhambasis>
		- xops: <n_eigstates x n_eigstates x num_cols>
		- exph: <n_eigstates x n_eigstates>
	*/
	DLLEXPORT int MakeOperatorsC(double* mat, int num_rows, int num_cols, int nhambasis, int npotbasis, double sigma,
						double step, double mass, double* simt, double* xops, complex<double>* exph)
	{
		Matrix<double> matM(num_cols, num_rows, mat);

		tuple<Matrix<complex<double>>, vector<Matrix<double>>, Matrix<double>> results =
			MakeOperators(matM, nhambasis, npotbasis, sigma, step, mass);
		Matrix<complex<double>> exph_mat(get<0>(results));
		vector<Matrix<double>> xops_vec(get<1>(results));
		Matrix<double> simt_mat(get<2>(results));

		/*
		the copy operations below are designed to work correctly whether n_eigstates is equal to or
		less than nhambasis
		
		if n_eigstates is less than nhambasis, then some rows and/or columns of output arrays will
		not be written to and should be thrown away by calling code
		
		(this complexity is the price we pay for knowing the size of the output arrays before the
		function is called)
		*/

		int n_eigstates = exph_mat.rows();

		// the output array (simt, xops, exph) all have nhambasis number of columns.  we may not write
		// to all of them, but it's important to use nhambasis (not n_eigstates) as the row stride
		int row_stride = nhambasis;

		// copy transposed similarity matrix to output array
		assert(simt_mat.rows() == n_eigstates);
		assert(simt_mat.cols() == nhambasis);
		// simt_mat and simt both have nhambasis columns, so we can copy the first n_eigstates rows the
		// easy way (any extra rows will just not be written to)
		memcpy(simt, simt_mat.raw(), n_eigstates * nhambasis * sizeof(double));

		// copy Hamiltonian time-evolution exponentiation operator matrix to output array
		assert(exph_mat.rows() == n_eigstates);
		assert(exph_mat.cols() == n_eigstates);
		int out_idx;
		for (int row_idx = 0; row_idx < n_eigstates; row_idx++)
		{
			for (int col_idx = 0; col_idx < n_eigstates; col_idx++)
			{
				out_idx = row_stride * row_idx + col_idx;
				exph[out_idx] = exph_mat.get(col_idx, row_idx);
			} // end for each column
		} // end for each row

		// copy position-expectation operator matrices to output array
		assert(xops_vec.size() == num_cols);
		int dim_stride = nhambasis * nhambasis;
		for (int dim_idx = 0; dim_idx < num_cols; dim_idx++)
		{
			assert(xops_vec[dim_idx].rows() == n_eigstates);
			assert(xops_vec[dim_idx].cols() == n_eigstates);
			for (int row_idx = 0; row_idx < n_eigstates; row_idx++)
			{
				for (int col_idx = 0; col_idx < n_eigstates; col_idx++)
				{
					out_idx = dim_stride * dim_idx + row_stride * row_idx + col_idx;
					xops[out_idx] = xops_vec[dim_idx].get(col_idx, row_idx);
				} // end for each column
			} // end for each row
		} // end for each dimension

		return n_eigstates;
	} // end function MakeOperatorsC


	/*
	build new frames based on the latest current frame, basis rows, operators, and sigma
	
	inputs
	* new_frames: allocated 3-D array of new frames, which we fill in (<num_rows x num_cols x num_frames_to_build)
	* num_rows: number of rows (in new_frames and current_frame)
	* num_cols: number of columns (in new_frames, current_frame, basis_rows)
	* num_frames_to_build: number of new frames to build
	* current_frame: double array containing current/latest frame (<num_rows x num_cols>)
	* basis_rows: double array containing basis rows (<num_basis rows x num_cols>) from first frame (before any DQM
	  evolution has taken place)
	* num_basis_rows: number of basis rows
	* simt: transpose of similarity matrix (<num_basis_vecs x num_basis_rows) -- used to to convert from representation
	  in the 'raw' basis of basis rows to representation in the orthonormal basis of eigenstates
	* num_basis_vecs: number of orthogonal basis vectors (may be smaller than num_basis_rows)
	* xops: 3-D array of position-expectation operator matrices (<num_basis_vecs x num_basis_vecs x num_cols>)
	* exph: Hamiltonian time-evolution exponentiation operator matrix (<num_basis_vecs x num_basis_vecs>)
	* sigma: value of sigma (for coherence, must be the same value that was used to build the operators)
	* stoppingthreshold: threshold for deciding when a row has stopped moving (see BuildFramesAuto for details)

	output
	* none	
	*/
	DLLEXPORT void BuildFramesAutoC(double* new_frames, int num_rows, int num_cols, int num_frames_to_build, double* current_frame,
							double* basis_rows, int num_basis_rows, double* simt, int num_basis_vecs, double* xops,
							complex<double>* exph, double sigma, double stopping_threshold)
	{
		assert(num_rows > 0);
		assert(num_cols> 0);
		assert(num_frames_to_build > 0);
		assert(num_basis_rows > 0);
		assert(num_basis_vecs <= num_basis_rows);
		assert(sigma > 0);

		Array3D all_frames_arr(num_cols, num_rows, num_frames_to_build + 1);
		// put current frame into first dim-3 slice of all_frames
		memcpy(all_frames_arr.raw(), current_frame, num_rows * num_cols * sizeof(double));
		
		Matrix<double> basis_rows_mat(num_cols, num_basis_rows, basis_rows);
		
		Matrix<double> simt_mat(num_basis_rows, num_basis_vecs, simt);

		Matrix<complex<double>> exph_mat(num_basis_vecs, num_basis_vecs, exph);
		
		// build vector of position-expectation matrices
		vector<Matrix<double>> xops_vec;
		xops_vec.reserve(num_cols); // note: we can't do this in the vector constructor because Matrix doesn't have a default constructor
		int dim_stride = num_basis_vecs * num_basis_vecs;
		for (int dim_idx = 0; dim_idx < num_cols; dim_idx++)
		{
			// note: it's okay that these matrices don't own their memory, since we're never going to change them
			Matrix<double> xop(num_basis_vecs, num_basis_vecs, xops + dim_idx * dim_stride);
			xops_vec.push_back(xop);
		} // end for each dimension/column
	
		BuildFramesAuto(all_frames_arr, basis_rows_mat, exph_mat, xops_vec, simt_mat, sigma, stopping_threshold);

		// copy new frames to output array
		int frame_stride = num_rows * num_cols;
		int row_stride = num_cols;
		memcpy(new_frames, all_frames_arr.raw() + num_rows * num_cols, num_rows * num_cols * num_frames_to_build * sizeof(double));
	} // end function BuildFramesAutoC

} // end extern "C"
