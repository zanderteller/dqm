#ifndef BASIS_H_
#define BASIS_H_

void ChooseBasisByDistance(double* rows, int num_rows, int num_cols, int basis_size, int* basis_row_nums, int first_basis_row_num);

void BuildOverlaps(double sigma, double* basis_rows, double* other_rows, int num_basis_rows, int num_other_rows, int num_cols, double* overlaps);

#endif //BASIS_H_
