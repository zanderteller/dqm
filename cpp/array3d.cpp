#include "array3d.h"

Array3D::Array3D(int xsize, int ysize, int zsize) :
xsize(xsize),
ysize(ysize),
zsize(zsize),
ystride(xsize),
zstride(ysize*xsize),
data_owned(true)
{
	data = new double[xsize * ysize * zsize];
}

Array3D::Array3D(int xsize, int ysize, int zsize, double *data) :
xsize(xsize),
ysize(ysize),
zsize(zsize),
ystride(xsize),
zstride(ysize*xsize),
data_owned(false),
data(data)
{
}

Array3D::~Array3D()
{
	if (data_owned)
	{
		delete[] data;
	}
}

double Array3D::get(int x, int y, int z) const
{
	return data[x + y * ystride + z * zstride];
}

void Array3D::set(int x, int y, int z, double value)
{
	data[x + y * ystride + z * zstride] = value;
}

Array3D Array3D::view(int xstart, int xsize, int ystart, int ysize, int zstart, int zsize)
{
	Array3D v(xsize, ysize, zsize, data + xstart + ystart * ystride + zstart * zstride);
	v.ystride = ystride;
	v.zstride = zstride;
	return v;
}

Matrix<double> Array3D::view2D(int xstart, int xsize, int ystart, int ysize, int zstart)
{
	return Matrix<double>(xsize, ysize, data + xstart + ystart * ystride + zstart * zstride);
}

// returns a new Array3D with a copy of all columns, selected rows, and all slices in dim 3
Array3D Array3D::get_rows(vector<int> &row_nums)
{
	Array3D v(xsize, (int)row_nums.size(), zsize);
	for (int z = 0; z < zsize; z++)
	{
		for (int y = 0; y < row_nums.size(); y++)
		{
			for (int x = 0; x < xsize; x++)
			{
				v.set(x, y, z, get(x, row_nums[y], z));
			}
		}
	}
	return v;
}

// returns a new matrix with a copy of all columns and selected rows from the single specified slice (z) in dim 3
Matrix<double> Array3D::get_rows(vector<int> &row_nums, int z)
{
	Matrix<double> m(xsize, (int)row_nums.size());
	for (int y = 0; y < row_nums.size(); y++)
	{
		for (int x = 0; x < xsize; x++)
		{
			m.set(x, y, get(x, row_nums[y], z));
		}
	}
	return m;
}
