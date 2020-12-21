#ifndef MAT_OP
#define MAT_OP

int random_uniform_initialize(double* m, int M, int N, double lower, double upper);
int naive_multiply_sequential(double* A, double* B, double* C, int M, int K, int N);
int naive_multiply_omp(double* A, double* B, double* C, int M, int K, int N);
void print_matrix(const double* mat, int M, int N);
void print_matrix_big(const double* mat, int M, int N, int include);
double max_diff(const double* mat1, const double* mat2, int M, int N);

#endif