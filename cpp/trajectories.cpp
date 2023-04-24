#include "array3d.h"
#include "matrix.h"
#include <complex>
#include <cmath>
#include <vector>
using std::vector;
using std::complex;

const int BATCH_SIZE = 5000; // number of data rows per batch

// 2FIX: REPLACE WITH A CALL TO A GENERALIZED VERSION OF BUILDOVERLAPS?
Matrix<double> VecsToEvolve(Matrix<double> &basis_rows, Matrix<double> &current_rows, double sigma)
{
	double den=1/4.0/sigma/sigma;
	Matrix<double> Vout(current_rows.rows(), basis_rows.rows());
	
	#pragma omp parallel for shared(basis_rows, current_rows, sigma, Vout)
	for (int i = 0; i < basis_rows.rows(); i++)
	{
		for (int j = 0; j < current_rows.rows(); j++)
		{
			double sq_dist = 0;
			for (int k = 0; k < basis_rows.cols(); k++)
			{
				double diff = basis_rows.get(k, i) - current_rows.get(k, j);
				sq_dist += diff * diff;
			}
			double overlap = exp(-sq_dist * den);
			overlap = overlap < 1e-12 ? 0 : overlap;
			Vout.set(j, i, overlap);
		}
	}
	return Vout;
}

/*
	GetXVals calculates and returns the expected position values, in all dimensions, for the eigenbasis states that are the columns of V

	Returned matrix is < number of states/points x number of dimensions >

	notes
	* the code has been optimized for execution speed.  (most notably, we receive xopvec in flattened 1D
	  form, which speeds things up a lot -- see also FlattenXops, where the flattening takes place.)
*/
Matrix<double> GetXVals(Matrix<complex<double>> &V, double *xopvec, int num_data_dims)
{
	// note: this is the number of orthogonal basis vectors, which may be smaller than the number of basis rows
	int num_basis_vecs = V.rows();

	Matrix<double> Xfinal(num_data_dims, V.cols());

	// loop in parallel over data rows
	#pragma omp parallel for shared(V,xopvec,num_data_dims,num_basis_vecs,Xfinal)
	for (int data_row_idx = 0; data_row_idx < V.cols(); data_row_idx++)
	{
		// initialize new data row with zeros
		double *new_data_row = new double[num_data_dims];
		for (int dim_idx = 0; dim_idx < num_data_dims; dim_idx++) // loop over data dimensions
		{
			new_data_row[dim_idx] = 0.0;
		}
		// get basis expansion for this data row
		complex<double> *data_row_basis_expansion = new complex<double>[num_basis_vecs];
		for (int basis_idx = 0; basis_idx < num_basis_vecs; basis_idx++) // loop over basis rows
		{
			data_row_basis_expansion[basis_idx] = V.get(data_row_idx, basis_idx);
		}
		complex<double> c1, c2;
		double c1real, c1imag, temp;
		int xopidx = 0;
		for (int basis_idx1 = 0; basis_idx1 < num_basis_vecs; basis_idx1++) // loop over all basis rows
		{
			c1 = data_row_basis_expansion[basis_idx1];
			c1real = c1.real();
			c1imag = c1.imag();

			// note: we only loop over the lower triangle of the basis-basis matrix, making use of the fact
			// that it is symmetric
			for (int basis_idx2 = 0; basis_idx2 <= basis_idx1; basis_idx2++)
			{
				c2 = data_row_basis_expansion[basis_idx2];
				temp = c1real * c2.real() + c1imag * c2.imag();
				if (basis_idx2 != basis_idx1)
				{
					temp *= 2.0; // this is how we include the upper triangle of the symmetric basis-basis matrix
				} // end if (not on diagonal of basis-basis matrix)
				for (int dim_idx = 0; dim_idx < num_data_dims; dim_idx++) // loop over data dimensions
				{
					// note: we receive xopvec in the appropriately ordererd 1D form (see FlattenXops), which
					// allows us to iterate over its values in order
					new_data_row[dim_idx] += temp * xopvec[xopidx++];
				} // end for (each data dimension)
			} // end for (each lower-triangle basis row)
		} // end for (each basis row)

		// put the new data row into the final data matrix
		for (int dim_idx = 0; dim_idx < num_data_dims; dim_idx++) // loop over data_dimensions
		{
			Xfinal.set(dim_idx, data_row_idx, new_data_row[dim_idx]);
		} // end for (each data dimension)

		delete[] new_data_row;
		delete[] data_row_basis_expansion;

	} // end for (each data row being processed)

	return Xfinal;
} // end GetXVals

// L2-normalize a complex-valued vector
void normalizepsi(Matrix<complex<double>>& psi)
{
	Matrix<complex<double>> retval(psi.cols(), psi.rows());
	int rdimpsi(psi.rows()), cdimpsi(psi.cols());
	
	#pragma omp parallel for shared(rdimpsi,cdimpsi,psi,retval)
	for (int i = 0; i < cdimpsi; i++)
	{
		double temp = 0.0;

		for (int j = 0; j < rdimpsi; j++)
		{
			complex<double> h1 = psi.get(i, j);
			temp = temp + h1.real() * h1.real() + h1.imag() * h1.imag();
		}
		if (temp > 1e-10)
			temp = 1 / sqrt(temp);
		else
			temp = 1.0;

		for (int k = 0; k < rdimpsi; k++)
			psi.set(i, k, psi.get(i, k) * temp);
	} // end for each column/state
} // end function normalizepsi

// 2FIX: IT'S AWKWARD AND A BIT DANGEROUS THAT THIS FUNCTION ALLOCATES MEMORY AND EXPECTS CALLING CODE TO DELETE IT.
// IT'S CURRENTLY SET UP THIS WAY BECAUSE I LIKE ENCAPSULATING HERE THE CALCULATION OF HOW MUCH MEMORY TO ALLOCATE.
double* FlattenXops(vector<Matrix<double>> &xops)
{
	/*
	For increased computation speed, we flatten xops into a 1D array, xopvec.

	Notes
	- We only put the lower triangle of each operator matrix into the array, making use
	  of the fact that the operator matrices are all symmetric.
	- The loops here must exactly match the corresponding loops in GetXVals, so that
	  iterating over the values of xopvec in order will give the correct result there.
	  (This obviously makes the code brittle, which is a tradoff for speed.)

	IMPORTANT NOTE: xopvec must be deleted by calling code to avoid a memory leak
	*/
	
	int num_data_dims = xops.size();
	
	// every entry in xops is a symmetric square matrix, all with size <num_basis_vecs x num_basis_vecs> (where
	// num_basis_vecs, the number of orthogonal basis vectors, may be smaller than nHambasis, the number of basis rows)
	int num_basis_vecs = xops[0].rows();

	double *xopvec = new double[num_data_dims * num_basis_vecs * (num_basis_vecs + 1) / 2];
	int xopidx = 0;
	for (int basis_idx1 = 0; basis_idx1 < num_basis_vecs; basis_idx1++) // loop over all basis vectors
	{
		for (int basis_idx2 = 0; basis_idx2 <= basis_idx1; basis_idx2++) // loop over lower-triangle basis vectors
		{
			for (int dim_idx = 0; dim_idx < num_data_dims; dim_idx++) // loop over data dimensions
			{
				xopvec[xopidx++] = xops[dim_idx].get(basis_idx1, basis_idx2);
			}
		}
	}

	return xopvec;
} // end function FlattenXops

// return a boolean representing whether a given row/point/sample has stopped moving (acorrding to the given stopping threshold)
bool RowHasStopped(Array3D &all_frames, int row_idx, int frame_idx, double sq_stopping_threshold)
{
	double sq_dist = 0;
	double diff;
	for (int col_idx = 0; col_idx < all_frames.cols(); col_idx++)
	{
		diff = all_frames.get(col_idx, row_idx, frame_idx) - all_frames.get(col_idx, row_idx, frame_idx - 1);
		sq_dist += diff * diff;
	}
	return sq_dist < sq_stopping_threshold;
} // end function RowHasStopped

// return a vector of row nums for a batch (for evolving rows in batches)
vector<int> BatchRowNums(int num_rows, bool* row_stopped, int batch_start_idx, int batch_size)
{
	vector<int> batch_row_nums;
	int evolving_idx = -1; // index for the still-evolving rows
	for (int row_idx = 0; row_idx < num_rows; row_idx++)
	{
		if (row_stopped[row_idx])
			continue;
		evolving_idx++;
		if (evolving_idx >= batch_start_idx)
			batch_row_nums.push_back(row_idx);
		if (batch_row_nums.size() == batch_size)
			break;
	} // for each row (building batch row nums)

	return batch_row_nums;
} // end function BatchRowNums

void BuildFramesAuto(Array3D &all_frames, Matrix<double> &basis_rows, Matrix<complex<double>> &exph, vector<Matrix<double> > &xops,
						Matrix<double> &simdag, double sigma, double stopping_threshold)
{
	int num_rows = all_frames.rows();
	int num_cols = all_frames.cols();
	int num_frames = all_frames.depth();

	assert(basis_rows.cols() == num_cols); // number of columns must be consistent
	assert(sigma > 0); // sigma must be positive

	// square the stopping threshold once to avoid having to take the square root for every norm
	double sq_stopping_threshold = stopping_threshold * stopping_threshold;

	double *xopvec = FlattenXops(xops); // IMPORTANT: remember to delete xopvec below

	// create combined operator
	Matrix<complex<double>> combo_op(exph * simdag);

	// track which rows are still evolving and which have stopped
	int num_evolving = num_rows;
	bool* row_stopped = new bool[num_rows];
	for (int i = 0; i < num_rows; i++)
		row_stopped[i] = false;

	// frame 0 is already filled in -- build forward from there
	int current_frame_idx, num_batches, batch_start_idx, actual_batch_size, evolving_idx, evolving_row_idx, batch_row_idx;
	for (int new_frame_idx = 1; new_frame_idx < num_frames; new_frame_idx++)
	{
		current_frame_idx = new_frame_idx - 1;

		if (num_evolving > 0)
		{
			// calculate current number of batches for the rows we're still evolving
			num_batches = int(ceil(double(num_evolving) / double(BATCH_SIZE)));

			for (int batch_idx = 0; batch_idx < num_batches; batch_idx++)
			{
				batch_start_idx = BATCH_SIZE * batch_idx;
				actual_batch_size = batch_idx == (num_batches - 1) ? num_evolving - batch_start_idx : BATCH_SIZE;

				// set up vector of row numbers for this batch
				vector<int> batch_row_nums(BatchRowNums(num_rows, row_stopped, batch_start_idx, actual_batch_size));
				Matrix<double> batch_rows(all_frames.get_rows(batch_row_nums, current_frame_idx));

				// evolve the batch rows

				// get basis-overlap vectors for each row
				Matrix<double> overlap_vecs(VecsToEvolve(basis_rows, batch_rows, sigma));

				// apply combined operator to get evolved states
				Matrix<complex<double>> psi(combo_op * overlap_vecs);

				// normalize evolved states
				// note: it's mathematically equivalent, and perhaps conceptually clearer, to normalize the state vector *before* evolving it.
				// in the 'Understanding DQM' whitepaper, normalization is presented as happening before application of the evolution operator.
				normalizepsi(psi);

				// get new expected position for the evolved state of each row
				Matrix<double> new_positions(GetXVals(psi, xopvec, num_cols));

				// copy evolved positions to new frame in final array
				for (int y = 0; y < new_positions.rows(); y++)
				{
					for (int x = 0; x < num_cols; x++)
					{
						all_frames.set(x, batch_row_nums[y], new_frame_idx, new_positions.get(x, y));
					}
				}

			} // end for each batch
		} // if any rows still evolving

		// mark newly stopped rows as stopped, and copy positions for already stopped rows to new frame
		for (int row_idx = 0; row_idx < num_rows; row_idx++)
		{
			if (row_stopped[row_idx])
			{
				for (int col_idx = 0; col_idx < num_cols; col_idx++)
					all_frames.set(col_idx, row_idx, new_frame_idx, all_frames.get(col_idx, row_idx, current_frame_idx));
			}
			else if (RowHasStopped(all_frames, row_idx, new_frame_idx, sq_stopping_threshold))
			{
					row_stopped[row_idx] = true;
					num_evolving--;
			} // end if/else (row already stopped or not)
		} // end for (dealing with stopped rows)

	} // end for (each new frame)

	delete[] xopvec;
	delete[] row_stopped;

} // end function BuildFramesAuto
