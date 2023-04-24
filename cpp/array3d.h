#ifndef MATRIX3D_H_
#define MATRIX3D_H_

#include "matrix.h"
#include <vector>

using std::vector;

/**
 * Represents a 3-dimmensional array of doubles in row-major order
 */
class Array3D
{
	double *data;
	int    xsize, ysize, zsize;
	int    ystride, zstride;
	bool   data_owned;
public:
	//normal constructor
	//xsize specifies the number of columns
	//ysize specifies the number of rows
	//zsize specifies the third dimmension
	Array3D(int xsize, int ysize, int zsize);
	//this contructor is useful for initializing an Array3D wrapper for Maple data
	//also used to implement the view method
	Array3D(int xsize, int ysize, int zsize, double *data);
	~Array3D();
	
	double *raw() { return data; }
	double const *raw() const { return data; }
	
	//fetch a single element from the array
	double get(int x, int y, int z) const;
	//set an element in the array
	void set(int x, int y, int z, double value);
	//these methods return the sizes of the individual dimmensions
	int rows() const { return ysize; }
	int cols() const { return xsize; }
	int depth() const { return zsize; }
	//this method returns a zero-copy view of a subset of the array
	//NOTE: the lifetime of the source array must be >= the lifetime of the view
	Array3D view(int xstart, int xsize, int ystart, int ysize, int zstart, int zsize);
	//Similar to view, but returns a matrix, useful for treating an Array3D like an array of matrices
	Matrix<double> view2D(int xstart, int xsize, int ystart, int ysize, int zstart);
	// returns a new Array3D with a copy of all columns, selected rows, and all slices in dim 3
	Array3D get_rows(vector<int> &row_nums);
	// returns a new matrix with a copy of all columns and selected rows from the single specified slice (z) in dim 3
	Matrix<double> get_rows(vector<int> &row_nums, int z);
};

#endif //MATRIX3D_H_
