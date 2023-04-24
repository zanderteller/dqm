#ifndef MATRIX2D_H_
#define MATRIX2D_H_
#include <cassert>
#include <vector>

using std::vector;

/**
 * Represents a real or complex matrix in row-major order
 */
template <class T>
class Matrix
{
	T *data;
	int    xsize, ysize;
	int    ystride;
	bool   data_owned;
public:
	//normal constructor
	//xsize specifies the number of columns
	//ysize specifies the number of rows
	Matrix(int xsize, int ysize);
	//this contructor is useful for initializing an Matrix wrapper for Maple data
	//also used to implement the view method
	Matrix(int xsize, int ysize, T *data);
	//copy constructor
	Matrix(Matrix const &src);
	~Matrix();
	
	//fetches a single element from the matrix
	T get(int x, int y) const;
	T *raw() { return data; }
	T const *raw() const { return data; }
	//sets a single element in the matrix
	void set(int x, int y, T value);
	//these methods return the size of individual matrix dimmensions
	int rows() const { return ysize; }
	int cols() const { return xsize; }
	
	//this method returns a zero-copy view of a subset of the matrix
	//NOTE: the lifetime of the source matrix must be >= the lifetime of the view
	Matrix view(int xstart, int xsize, int ystart, int ysize);
	const Matrix view(int xstart, int xsize, int ystart, int ysize) const;

	// return the transpose of the matrix
	Matrix<T> transpose() const;

	//matrix multiplication with an explicit result reference
	template <class U>
	void mult(Matrix<U> const &right, Matrix<T> &result) const;
	
	//matrix multiplication in convenient operator format
	template <class U>
	Matrix<T> operator*(Matrix<U> const &right) const;

	// scalar multiplication
	template <class U>
	Matrix<T> operator*(U const right) const;

	// scalar division
	template <class U>
	Matrix<T> operator/(U const right) const;

	// scalar addition
	template <class U>
	Matrix<T> operator+(U const right) const;

	// matrix addition (matrices must have identical dimensions)
	template <class U>
	Matrix<T> operator+(Matrix<U> const& right) const;

	// returns a new Matrix with a copy of all columns and selected rows
	Matrix<T> get_rows(vector<int> &rows);

};

#include <cstring>

template <class T>
Matrix<T>::Matrix(int xsize, int ysize) :
xsize(xsize),
ysize(ysize),
ystride(xsize),
data_owned(true)
{
	data = new T[xsize * ysize];
}

template <class T>
Matrix<T>::Matrix(int xsize, int ysize, T *data) :
xsize(xsize),
ysize(ysize),
ystride(xsize),
data_owned(false),
data(data)
{
}

template <class T>
Matrix<T>::Matrix(Matrix const &src) :
xsize(src.xsize),
ysize(src.ysize),
ystride(src.xsize),
data_owned(true)
{
	data = new T[xsize * ysize];
	if (src.ystride == src.xsize) {
		memcpy(data, src.data, xsize * ysize * sizeof(T));
	} else {
		T *cur = data;
		for (int y = 0; y < ysize; y++)
		{
			for (int x = 0; x < xsize; x++)
			{
				*(cur++) = src.get(x, y);
			}
		}
	}
}

template <class T>
Matrix<T>::~Matrix()
{
	if (data_owned)
	{
		delete[] data;
	}
}

template <class T>
T Matrix<T>::get(int x, int y) const
{
	return data[x + y * ystride];
}

template <class T>
void Matrix<T>::set(int x, int y, T value)
{
	data[x + y * ystride] = value;
}

template <class T>
Matrix<T> Matrix<T>::view(int xstart, int xsize, int ystart, int ysize)
{
	Matrix<T> v(xsize, ysize, data + xstart + ystart * ystride);
	v.ystride = ystride;
	return v;
}

template <class T>
const Matrix<T> Matrix<T>::view(int xstart, int xsize, int ystart, int ysize) const
{
	Matrix<T> v(xsize, ysize, data + xstart + ystart * ystride);
	v.ystride = ystride;
	return v;
}

template <class T>
Matrix<T> Matrix<T>::transpose() const
{
	Matrix<T> mat_t(rows(), cols());
	#pragma omp parallel for shared(mat_t)
	for (int col_idx = 0; col_idx < cols(); col_idx++)
	{
		for (int row_idx = 0; row_idx < rows(); row_idx++)
		{
			mat_t.set(row_idx, col_idx, get(col_idx, row_idx));
		}
	}
	return mat_t;
}

template <class T>
template <class U>
void Matrix<T>::mult(Matrix<U> const &right, Matrix<T> &result) const
{
	#pragma omp parallel for shared(result)
	for (int dstx = 0; dstx < result.cols(); dstx++)
	{
		for (int dsty = 0; dsty < result.rows(); dsty++)
		{
			T accum{};
			for (int src = 0; src < cols(); src++)
			{
				accum += get(src, dsty) * right.get(dstx, src);
			}
			result.set(dstx, dsty, accum);
		}
	}
}

template <class T>
template <class U>
Matrix<T> Matrix<T>::operator*(Matrix<U> const &right) const
{
	assert(right.rows() == cols());
	Matrix<T> result(right.cols(), rows());
	mult(right, result);
	return result;
}

// scalar multiplication
template <class T>
template <class U>
Matrix<T> Matrix<T>::operator*(U const right) const
{
	Matrix<T> result(cols(), rows());
	#pragma omp parallel for shared(result)
	for (int col_idx = 0; col_idx < cols(); col_idx++)
	{
		for (int row_idx = 0; row_idx < rows(); row_idx++)
		{
			result.set(col_idx, row_idx, get(col_idx, row_idx) * right);
		}
	}
	return result;
} // end operator (scalar multiplication)

// scalar division
template <class T>
template <class U>
Matrix<T> Matrix<T>::operator/(U const right) const
{
	Matrix<T> result(cols(), rows());
	#pragma omp parallel for shared(result)
	for (int col_idx = 0; col_idx < cols(); col_idx++)
	{
		for (int row_idx = 0; row_idx < rows(); row_idx++)
		{
			result.set(col_idx, row_idx, get(col_idx, row_idx) / right);
		}
	}
	return result;
} // end operator (scalar division)

// scalar addition
template <class T>
template <class U>
Matrix<T> Matrix<T>::operator+(U const right) const
{
	Matrix<T> result(cols(), rows());
	#pragma omp parallel for shared(result)
	for (int col_idx = 0; col_idx < cols(); col_idx++)
	{
		for (int row_idx = 0; row_idx < rows(); row_idx++)
		{
			result.set(col_idx, row_idx, get(col_idx, row_idx) + right);
		}
	}
	return result;
} // end operator (scalar multiplication)

// matrix addition (matrices must have identical dimensions)
template <class T>
template <class U>
Matrix<T> Matrix<T>::operator+(Matrix<U> const& right) const
{
	assert(right.rows() == rows() && right.cols() == cols());
	Matrix<T> result(cols(), rows());
	#pragma omp parallel for shared(result)
	for (int col_idx = 0; col_idx < cols(); col_idx++)
	{
		for (int row_idx = 0; row_idx < rows(); row_idx++)
		{
			result.set(col_idx, row_idx, get(col_idx, row_idx) + right.get(col_idx, row_idx));
		}
	}
	return result;
} // end operator (matrix addition)

// return a new matrix with the specified subset of rows
template <class T>
Matrix<T> Matrix<T>::get_rows(vector<int> &rows)
{
	Matrix<T> result(xsize, rows.size());
	for (int y = 0; y < rows.size(); y++)
	{
		for (int x = 0; x < xsize; x++)
		{
			result.set(x, y, get(x, rows[y]));
		}
	}
	return result;
} // end method get_rows

#endif //MATRIX2D_H_