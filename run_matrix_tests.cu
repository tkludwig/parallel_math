/*goal: compare performance of a few sequential & parallel implementations of matrix multiplication*/

#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <omp.h>
#include <chrono>
#include <random>
#include <iomanip>

#include "matrix_operations.cuh"

int main()
{
	int M = 8;
	int N = 8;
	double mat1[64];
	int init_failed = random_uniform_initialize(mat1, M, N, 0.0, 10.5);
	print_matrix(mat1, 8, 8);
	print_matrix_big(mat1, 8, 8, 3);

	std::cout << "test sequential naive multiply:" << std::endl;
	double A[4] = { 2., 0., 0., 3. };
	double B[4] = { 2., 0., 1., 2. };
	double C[4];
	naive_multiply_sequential(A, B, C, 2, 2, 2);
	print_matrix(C, 2, 2);
}