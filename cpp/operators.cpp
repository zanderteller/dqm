#include <cstdlib>
#include <vector>
#include <complex>
#include <tuple>
#include <assert.h>
#include "matrix.h"
#include "utils.h"
#include "operators.h"

using std::vector;
using std::tuple;
using std::make_tuple;
using std::get;
using std::complex;
using namespace std::complex_literals;

/********************

This file defines MakeOperators (exported in dqm_python.cpp) and supporting functions.

********************/

/*
given a matrix, create a square matrix of Gaussian overlaps between each pair of rows

(2FIX: there's a bit of naming confusion between MakeOverlaps here and BuildOverlaps in basis.cpp)

inputs
* mat: data matrix
* sigma: value of sigma (width of Gaussians)

output
* square matrix of pairwise Gaussian overlaps (inner products)
*/
Matrix<double> MakeOverlaps(Matrix<double>& mat, double sigma)
{
	int nrows = mat.rows();
	int ncols = mat.cols();
	Matrix<double> overlaps(nrows, nrows);

	double sigma_factor = 4 * sigma * sigma; // do this calculation just once

	// traverse only the upper triangle of the overlaps matrix (since it's symmetric)
	// 2FIX: the omp pragma isn't splitting the work up evenly here
	#pragma omp parallel for shared(mat, sigma_factor, nrows, ncols, overlaps)
	for (int row_idx_1 = 0; row_idx_1 < nrows; row_idx_1++)
	{
		double sq_dist, diff, overlap;
		for (int row_idx_2 = row_idx_1; row_idx_2 < nrows; row_idx_2++)
		{
			if (row_idx_1 == row_idx_2) {
				overlaps.set(row_idx_1, row_idx_2, 1); // overlap between a row and itself is always 1
			}
			else {
				// calculate overlap between two rows
				sq_dist = 0;
				for (int col_idx = 0; col_idx < ncols; col_idx++)
				{
					diff = mat.get(col_idx, row_idx_1) - mat.get(col_idx, row_idx_2);
					sq_dist += diff * diff;
				} // end for each column
				overlap = exp(-sq_dist / sigma_factor);
				if (overlap < 1e-12) {
					overlap = 0;
				}
				// fill in both upper and lower triangles of the overlaps matrix
				overlaps.set(row_idx_1, row_idx_2, overlap);
				overlaps.set(row_idx_2, row_idx_1, overlap);
			} // end if/else (rows are same or different)
		} // end for each row in upper triangle
	} // end for each row
	return overlaps;
} // end function MakeOverlaps

/*
given matrices mat1 <rows1 x cols> and mat2 <rows2 x cols>, create a <rows1 x rows2> matrix of Gaussian overlaps
between the Cartesian product of rows from mat1 and mat2

(2FIX: there's a bit of naming confusion between MakeOverlaps here and BuildOverlaps in basis.cpp)

inputs
* mat1: data matrix 1
* mat2: data matrix 2
* sigma: value of sigma (width of Gaussians)

output
* <rows1 x rows2> matrix of pairwise Gaussian overlaps (inner products)
*/
Matrix<double> MakeOverlaps(Matrix<double>& mat1, Matrix<double>& mat2, double sigma)
{
	int nrows1 = mat1.rows();
	int nrows2 = mat2.rows();
	int ncols = mat1.cols();

	assert(ncols == mat2.cols());

	Matrix<double> overlaps(nrows2, nrows1);

	double sigma_factor = 4 * sigma * sigma; // do this calculation just once

	#pragma omp parallel for shared(mat1, mat2, sigma_factor, nrows1, nrows2, ncols, overlaps)
	for (int row1_idx = 0; row1_idx < nrows1; row1_idx++)
	{
		double sq_dist, diff, overlap;
		for (int row2_idx = 0; row2_idx < nrows2; row2_idx++)
		{
			// calculate overlap between two rows
			sq_dist = 0;
			for (int col_idx = 0; col_idx < ncols; col_idx++)
			{
				diff = mat1.get(col_idx, row1_idx) - mat2.get(col_idx, row2_idx);
				sq_dist += diff * diff;
			} // end for each column
			overlap = exp(-sq_dist / sigma_factor);
			if (overlap < 1e-12) {
				overlap = 0;
			}
			overlaps.set(row2_idx, row1_idx, overlap);
		} // end for each row in mat2
	} // end for each row in mat1
	return overlaps;
} // end function MakeOverlaps

/*
aggregate the contributions from each matrix row to the raw building blocks of the potential, which are
the 'pot' matrix and the 'psi' matrix, both of which are <nhambasis x nhambasis>.

note that the rows of mat and mat_ham_sub may be disjoint, which allows a distributed version of this function
to call this function on a partition of all rows in the original matrix.

inputs:
* mat: data matrix
* mat_ham_sub: matrix of basis rows
* sigma: value of sigma (width of Gaussians)

output:
* a tuple of <pot_mat, psi_mat>
*/
tuple<Matrix<double>, Matrix<double>> AggregatePotentialContributions(Matrix<double>& mat, Matrix<double>& mat_ham_sub, double sigma)
{
	int nhambasis = mat_ham_sub.rows();
	int nrows = mat.rows();
	int ncols = mat.cols();

	Matrix<double> pot_mat(nhambasis, nhambasis);
	Matrix<double> psi_mat(nhambasis, nhambasis);

	double den = 2 * sigma * sigma;
	
	// in order to get OMP to split up the work evenly in the main loop below, we first flatten
	// the nested for loops that traverse the upper triangle of the overlaps matrix
	int n = nhambasis * (nhambasis + 1) / 2;
	vector<int> overlap_row_idx_1_vec(n);
	vector<int> overlap_row_idx_2_vec(n);
	int idx = -1;
	for (int overlap_row_idx_1 = 0; overlap_row_idx_1 < nhambasis; overlap_row_idx_1++) {
		for (int overlap_row_idx_2 = overlap_row_idx_1; overlap_row_idx_2 < nhambasis; overlap_row_idx_2++) {
			idx++;
			overlap_row_idx_1_vec[idx] = overlap_row_idx_1;
			overlap_row_idx_2_vec[idx] = overlap_row_idx_2;
		}
	}

	// now run the single flattened for loop, where each unique entry (overlap_row_idx_1, overlap_row_idx_2) in the
	// 'pot' and 'psi' matrices (upper triangle only) is its own parallel job
	#pragma omp parallel for shared(mat, mat_ham_sub, den, nrows, ncols, pot_mat, psi_mat, overlap_row_idx_1_vec, overlap_row_idx_2_vec)
	for (int idx = 0; idx < n; idx++) {
		int overlap_row_idx_1 = overlap_row_idx_1_vec[idx];
		int overlap_row_idx_2 = overlap_row_idx_2_vec[idx];

		double psi = 0.0;
		double pot = 0.0;
		double temppsi, temppot, tempdiffs, temp;

		// calculate mean row, just once, for these 2 basis rows
		Matrix<double> mean_row(ncols, 1);
		for (int col_idx = 0; col_idx < ncols; col_idx++) {
			temp = (mat_ham_sub.get(col_idx, overlap_row_idx_1) + mat_ham_sub.get(col_idx, overlap_row_idx_2)) / 2.0;
			mean_row.set(col_idx, 0, temp);
		}

		// loop over all rows -- each row will make its own contribution to the value of the potential
		// evaluated at the point defined by the mean row
		for (int row_idx = 0; row_idx < nrows; row_idx++) {
			tempdiffs = 0.0;
			for (int col_idx = 0; col_idx < ncols; col_idx++) {
				temp = mean_row.get(col_idx, 0) - mat.get(col_idx, row_idx);
				tempdiffs += temp * temp;
			}
			temppsi = exp(-tempdiffs / den);
			temppot = tempdiffs * temppsi;
			pot += temppot;
			psi += temppsi;
		}

		if (overlap_row_idx_1 == overlap_row_idx_2) {
			pot_mat.set(overlap_row_idx_2, overlap_row_idx_1, pot);
			psi_mat.set(overlap_row_idx_2, overlap_row_idx_1, psi);
		}
		else {
			pot_mat.set(overlap_row_idx_2, overlap_row_idx_1, pot);
			psi_mat.set(overlap_row_idx_2, overlap_row_idx_1, psi);

			pot_mat.set(overlap_row_idx_1, overlap_row_idx_2, pot);
			psi_mat.set(overlap_row_idx_1, overlap_row_idx_2, psi);
		}
	} // end for (each unique overlaps entry)

	return make_tuple(pot_mat, psi_mat);

} // end function AggregatePotentialContributions

/*
create the initial potential matrix, V0

new version (2021-07-03): when the total number of rows is large, aggregating the contribution to the potential
from each row is the really computationally expensive part.  this new version splits up the work differently so that the
aggregation step can happen simultaneously on multiple machines.

inputs
* mat: data matrix
* mat_ham_sub: matrix of basis rows
* overlaps: square matrix of basis-basis overlaps
* sigma: value of sigma (width of Gaussians)

output
* V0: square matrix, the same size as the overlaps matrix passed in.  each entry is the basis-basis
    overlap for that pair of basis rows times the potential evaluated at the midpoint between those
	2 basis rows.
*/
Matrix<double> MakeV0(Matrix<double>& mat, Matrix<double>& mat_ham_sub, Matrix<double>& overlaps, double sigma)
{
	tuple<Matrix<double>, Matrix<double>> results = AggregatePotentialContributions(mat, mat_ham_sub, sigma);
	Matrix<double> pot_mat(get<0>(results));
	Matrix<double> psi_mat(get<1>(results));

	int nhambasis = mat_ham_sub.rows(); // number of rows in the basis

	Matrix<double> V0(nhambasis, nhambasis);

	// in order to get OMP to split up the work evenly in the main loop below, we first flatten
	// the nested for loops that traverse the upper triangle of the overlaps matrix
	int n = nhambasis * (nhambasis + 1) / 2;
	vector<int> overlap_row_idx_1_vec(n);
	vector<int> overlap_row_idx_2_vec(n);
	int idx = -1;
	for (int overlap_row_idx_1 = 0; overlap_row_idx_1 < nhambasis; overlap_row_idx_1++) {
		for (int overlap_row_idx_2 = overlap_row_idx_1; overlap_row_idx_2 < nhambasis; overlap_row_idx_2++) {
			idx++;
			overlap_row_idx_1_vec[idx] = overlap_row_idx_1;
			overlap_row_idx_2_vec[idx] = overlap_row_idx_2;
		}
	}

	// now run the single flattened for loop, where each unique entry (overlap_row_idx_1, overlap_row_idx_2) in the
	// overlaps matrix (upper triangle only) is its own parallel job
	#pragma omp parallel for shared(overlaps, pot_mat, psi_mat, V0, overlap_row_idx_1_vec, overlap_row_idx_2_vec)
	for (int idx = 0; idx < n; idx++) {
		int overlap_row_idx_1 = overlap_row_idx_1_vec[idx];
		int overlap_row_idx_2 = overlap_row_idx_2_vec[idx];
		
		double v;
		double psi = psi_mat.get(overlap_row_idx_2, overlap_row_idx_1);
		if (psi == 0.0) {
			v = 0;
		}
		else {
			double pot = pot_mat.get(overlap_row_idx_2, overlap_row_idx_1);
			// multiply the overlap between these 2 rows by the potential evaluated at the midpoint between them
			v = overlaps.get(overlap_row_idx_2, overlap_row_idx_1) * pot / psi;
		}

		if (overlap_row_idx_1 == overlap_row_idx_2) {
			V0.set(overlap_row_idx_2, overlap_row_idx_1, v);
		}
		else {
			V0.set(overlap_row_idx_2, overlap_row_idx_1, v);
			V0.set(overlap_row_idx_1, overlap_row_idx_2, v);
		}
	} // end for (each unique overlaps entry)

	return V0;

} // end function MakeV0

/*
create the initial Hamiltonian operator matrix, H0

inputs
* mat_ham_sub: matrix of basis rows
* overlaps: square matrix of basis-basis overlaps
* sigma: value of sigma (width of Gaussians)

output
* H0: the initial Hamiltonian operator matrix
*/
Matrix<double> MakeH0(Matrix<double> mat_ham_sub, Matrix<double> overlaps)
{
	int nhambasis = mat_ham_sub.rows();
	int ncols = mat_ham_sub.cols();
	assert(nhambasis == overlaps.rows() && nhambasis == overlaps.cols());
	Matrix<double> H0(nhambasis, nhambasis);

	// only traverse the upper triangle of the symmetric basis-basis overlap matrix
	// 2FIX: the omp pragma isn't splitting the work up evenly here
	#pragma omp parallel for shared(mat_ham_sub, overlaps, nhambasis, ncols, H0)
	for (int overlap_row_idx_1 = 0; overlap_row_idx_1 < nhambasis; overlap_row_idx_1++)
	{
		double sq_dist, h;
		double delta;
		for (int overlap_row_idx_2 = overlap_row_idx_1; overlap_row_idx_2 < nhambasis; overlap_row_idx_2++)
		{
			if (overlap_row_idx_1 == overlap_row_idx_2) {
				H0.set(overlap_row_idx_1, overlap_row_idx_2, 0);
			}
			else {
				sq_dist = 0;
				for (int col_idx = 0; col_idx < ncols; col_idx++) {
					delta = mat_ham_sub.get(col_idx, overlap_row_idx_1) - mat_ham_sub.get(col_idx, overlap_row_idx_2);
					sq_dist += delta * delta;
				}
				h = -sq_dist * overlaps.get(overlap_row_idx_1, overlap_row_idx_2);
				H0.set(overlap_row_idx_1, overlap_row_idx_2, h);
				H0.set(overlap_row_idx_2, overlap_row_idx_1, h);
			} // end if/else (on the diagonal or not)
		} // end for (basis index 2 -- upper triangle only)
	} // end for (basis index 1)
	return H0;
} // end function MakeH0

/*
build and return the initial Hamiltonian matrix

inputs
* mat_ham_sub: matrix of basis rows
* overlaps: square matrix of basis-basis overlaps
* V0: potential matrix (see MakeV0)
* sim_t: tranpose of 'similarity' matrix (see MakeSim)
* sigma: sigma
* mass: mass

output
* htrunc: initial Hamiltonian matrix (trunc = 'truncated', meaning we're truncated to the subspace described by the basis eigenstates)
*/
Matrix<double> MakeH(Matrix<double>& mat_ham_sub, Matrix<double>& overlaps, Matrix<double>& V0, Matrix<double> sim_t,
	double sigma, double mass)
{
	Matrix<double> H0(MakeH0(mat_ham_sub, overlaps));

	double v0_coef, h0_coef;

	// V0 and H0 both have a factor of 1/sigma^2, which makes the whole system scale-invariant (if the raw data and sigma
	// are both multiplied by some arbitrary scale factor, the evolution will not change at all)
	v0_coef = 1 / (sigma * sigma);
	h0_coef = 1 / (mass * sigma * sigma);

	Matrix<double> H(V0 * v0_coef + H0 * h0_coef);

	// convert H to the orthogonal basis
	Matrix<double> H_orth(sim_t * H * sim_t.transpose());

	return H_orth;
} // end function MakeH

/*
create the position-expectation operator matrix for a single given dimension

inputs
* mat_ham_sub: matrix of basis rows
* overlaps: square matrix of basis-basis overlaps
* col_idx: number of the column/dimension for which we're building the position-expectation operator matrix

output
* X: the position-expectation operator matrix for the given dimension
*/
Matrix<double> MakeX(Matrix<double> mat_ham_sub, Matrix<double> overlaps, int col_idx)
{
	int nhambasis = mat_ham_sub.rows();

	Matrix<double> X(nhambasis, nhambasis);

	// only traverse the upper triangle of the symmetric basis-basis overlap matrix
	// 2FIX: the omp pragma isn't splitting the work up evenly here
	#pragma omp parallel for shared(mat_ham_sub, overlaps, col_idx, nhambasis, X)
	for (int overlap_row_idx_1 = 0; overlap_row_idx_1 < nhambasis; overlap_row_idx_1++)
	{
		double val;
		for (int overlap_row_idx_2 = overlap_row_idx_1; overlap_row_idx_2 < nhambasis; overlap_row_idx_2++)
		{
			if (overlap_row_idx_1 == overlap_row_idx_2) {
				// overlap of a row with itself is always 1, so this entry is just the coordinate of this basis row
				// in the given dimension
				X.set(overlap_row_idx_1, overlap_row_idx_2, mat_ham_sub.get(col_idx, overlap_row_idx_1));
			}
			else {
				// entry is the overlap between these 2 basis rows times the coordinate in the given dimension of the
				// midpoint between them
				val = overlaps.get(overlap_row_idx_1, overlap_row_idx_2) * (mat_ham_sub.get(col_idx, overlap_row_idx_1) + mat_ham_sub.get(col_idx, overlap_row_idx_2)) / 2;
				X.set(overlap_row_idx_1, overlap_row_idx_2, val);
				X.set(overlap_row_idx_2, overlap_row_idx_1, val);
			} // end if/else (on the diagonal or not)
		} // end for (basis index 2 -- upper triangle only)
	} // end for (basis index 1)
	return X;
} // end function MakeX

/*
build the Hamiltonian time-evolution exponentiation operator matrix

inputs
* h: the initial Hamiltonian operator matrix (see MakeH0)
* step: value of time step

output
* exph: the Hamiltonian time-evolution exponentiation operator matrix
*/
Matrix<complex<double>> MakeHIter(Matrix<double> h, double step)
{
	// the plan: diagonalize h, exponentiate it, and then undiagonalize it again

	int num_basis_vecs = h.rows();

	// get eigenvalues and eigenvectors for h
	Matrix<double> eig_vals(1, num_basis_vecs);
	Matrix<double> eig_vecs(num_basis_vecs, num_basis_vecs);
	EigValsVecs(h, eig_vals, eig_vecs);

	// subtract the first eigenvalue from each eigenvalue
	// note: taking the difference of each eigenvalue from the first eigenvalue is all that matters in evolution;
	// the energy splittings are just smaller if you take away the overall phase factor (i.e., the overall
	// ground-state energy).
	double first_eig_val = eig_vals.get(0, 0);
	for (int idx = 0; idx < num_basis_vecs; idx++)
	{
		eig_vals.set(0, idx, eig_vals.get(0, idx) - first_eig_val);
	}

	// create diagonal matrix where diagonal elements are exp(-i * step * adjusted_eigenvalue)
	Matrix<complex<double>> temp(num_basis_vecs, num_basis_vecs);
	#pragma omp parallel for shared(temp, eig_vals, num_basis_vecs, step)
	for (int idx1 = 0; idx1 < num_basis_vecs; idx1++)
	{
		complex<double> val;
		for (int idx2 = 0; idx2 < num_basis_vecs; idx2++)
		{
			if (idx1 == idx2) {
				val = exp(-1i * step * eig_vals.get(0, idx1));
			}
			else {
				val = 0;
			}
			temp.set(idx1, idx2, val);
		}
	}

	// convert eig_vecs to complex type (have to do this before multiplying)
	Matrix<complex<double>> eig_vecs_c(num_basis_vecs, num_basis_vecs);
	#pragma omp parallel for shared(eig_vecs_c, eig_vecs, num_basis_vecs)
	for (int row_idx = 0; row_idx < num_basis_vecs; row_idx++)
	{
		for (int col_idx = 0; col_idx < num_basis_vecs; col_idx++)
		{
			eig_vecs_c.set(col_idx, row_idx, eig_vecs.get(col_idx, row_idx));
		}
	}
	Matrix<complex<double>> eig_vecs_c_t(eig_vecs_c.transpose());

	// undiagonalize temp and return the result
	return eig_vecs_c * temp * eig_vecs_c_t;
} // end function MakeHIter

/*
make the 'sim' (for 'similarity') matrix that defines conversion between representation in the 'raw' basis of basis rows and
the orthonormal basis of eigenstates.

sim is the matrix of suitably normalized eigenvectors of the basis-basis overlap matrix, where "suitably normalized" means
that converting the overlaps matrix to the orthonormal basis produces the identity matrix:
		sim_t * overlaps * sim = I

inputs
* overlaps: square matrix of basis-basis overlaps

output
* sim: the similarity matrix.  note: sim may not be square.  it will be <num_eigen_vecs x num_basis_rows>, where the number
	of eigenvectors may be smaller than the number of basis rows, because we may have dropped eigenvectors where the eigenvalues
	were too small. (note: if the eigenvector calculations on the overlaps matrix fail to converge, which can happen when there
	are too many small values, we return a sim matrix with all zeros.)
*/
Matrix<double> MakeSim(Matrix<double> overlaps)
{
	// get eigenvalues and eigenvectors for the square basis-basis overlap matrix
	int nhambasis = overlaps.rows();
	Matrix<double> eig_vals_full(1, nhambasis);
	Matrix<double> eig_vecs_full(nhambasis, nhambasis);
	EigValsVecs(overlaps, eig_vals_full, eig_vecs_full);

	// if largest (last) eigenvalue is zero (meaning eigenvector calculations failed to converg), return
	// a sim matrix with all zeros
	if (eig_vals_full.get(0, nhambasis - 1) == 0) {
		Matrix<double> sim(nhambasis, nhambasis);
		for (int col_idx = 0; col_idx < nhambasis; col_idx++)
			for (int row_idx = 0; row_idx < nhambasis; row_idx++)
				sim.set(col_idx, row_idx, 0);
		return sim;
	}

	// eigenvalues are returned in increasing order, so we cut too-small states from the beginning of the vector
	int start_eig_idx = 0;
	while (eig_vals_full.get(0, start_eig_idx) < 1e-5) { // 2FIX: MOVE 1e-5 CONSTANT TO TOP OF FILE (OR MAKE IT A PARAMETER)
		start_eig_idx++;
	}
	int num_basis_vecs = nhambasis - start_eig_idx;
	// subselect to the big-enough eigenvalues and eigenvectors we're going to use
	Matrix<double> eig_vals(eig_vals_full.view(0, 1, start_eig_idx, num_basis_vecs));
	Matrix<double> eig_vecs(eig_vecs_full.view(start_eig_idx, num_basis_vecs, 0, nhambasis));
	Matrix<double> eig_vecs_t(eig_vecs.transpose());

	// the similarity matrix is the matrix of suitably normalized overlap eigenvectors
	Matrix<double> sim(num_basis_vecs, nhambasis);
	#pragma omp parallel for shared(sim, eig_vals, eig_vecs)
	for (int col_idx = 0; col_idx < sim.cols(); col_idx++) {
		double eig_val = eig_vals.get(0, col_idx);
		double mult = 1 / sqrt(eig_val);
		for (int row_idx = 0; row_idx < sim.rows(); row_idx++)
			sim.set(col_idx, row_idx, eig_vecs.get(col_idx, row_idx) * mult);
	}

	return sim;
} // end function MakeSim

/*
build and return the DQM operators

inputs
* mat: data matrix
* nhambasis: number of rows (starting from the first row) to use as the basis
* npotbasis: number of rows (starting from the first row) to use to build the potential
* sigma: value of sigma (width of Gaussians)
* step: value of time step
* mass: value of mass

output -- tuple of:
* eth: complex matrix -- the Hamiltonian time-evolution exponentiation operator matrix
* xopvec: vector of position-expectation matrices, one for each dimension
* sim_t: transpose of 'similarity' matrix -- used to to convert from representation in the 'raw'
    basis of basis rows to representation in the orthonormal basis of eigenstates
*/
tuple<Matrix<complex<double>>, vector<Matrix<double>>, Matrix<double>> MakeOperators(Matrix<double> mat, int nhambasis, int npotbasis,
																						double sigma, double step, double mass)
{
	int nrows = mat.rows();
	int ncols = mat.cols();

	assert(nhambasis <= nrows && npotbasis <= nrows);
	assert(nhambasis <= npotbasis);

	// get basis rows
	Matrix<double> mat_ham_sub(mat.view(0, ncols, 0, nhambasis));

	// calculate basis-basis-overlaps
	Matrix<double> overlaps(MakeOverlaps(mat_ham_sub, sigma));

	// create the 'similarity' matrix
	Matrix<double> sim(MakeSim(overlaps));
	Matrix<double> sim_t(sim.transpose());

	// create the position-expectation operators
	vector<Matrix<double>> xopvec;
	xopvec.reserve(ncols);
	for (int col_idx = 0; col_idx < ncols; col_idx++)
	{
		xopvec.push_back(sim_t * MakeX(mat_ham_sub, overlaps, col_idx) * sim);
	}

	// create the potential matrix
	Matrix<double> mat_for_pot(mat.view(0, ncols, 0, npotbasis));
	Matrix<double> V0(MakeV0(mat_for_pot, mat_ham_sub, overlaps, sigma));

	// create the initial Hamiltonian
	Matrix<double> H(MakeH(mat_ham_sub, overlaps, V0, sim_t, sigma, mass));

	// finally create the time-evolution exponentiation operator
	Matrix<complex<double>> eth = MakeHIter(H, step);

	return make_tuple(eth, xopvec, sim_t);
} // end function MakeOperators
