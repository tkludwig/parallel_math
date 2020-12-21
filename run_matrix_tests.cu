/*goal: compare performance of a few sequential & parallel implementations of matrix multiplication*/

#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <omp.h>
#include <chrono>
#include <random>
#include <iomanip>
#include <armadillo>

#include "matrix_operations.cuh"

int main()
{
	std::cout << "Armadillo version: " << arma::arma_version::as_string() << std::endl;
	const int M = 79;
	const int N = 79;
	double mat1[M*N];
	double mat2[M*N];
	double mat3[M*N];
	int init_failed = random_uniform_initialize(mat1, M, N, 0.0, 0.5);

	std::cout << "test sequential naive multiply:" << std::endl;
	double A[4] = { 2., 0., 0., 3. };
	double B[4] = { 2., 0., 1., 2. };
	double C[4];
	naive_multiply_sequential(A, B, C, 2, 2, 2);
	print_matrix(C, 2, 2);

	std::cout << "test omp naive multiply:" << std::endl;
	auto start = std::chrono::high_resolution_clock::now();
	naive_multiply_omp(mat1, mat1, mat2, M, N, N);
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	std::cout << "naive_multiple_omp on size " << M << " took " << duration.count() << " ms" << std::endl;

	start = std::chrono::high_resolution_clock::now();
	naive_multiply_sequential(mat1, mat1, mat3, M, N, N);
	end = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	std::cout << "naive_multiple_sequential on size " << M << " took " << duration.count() << " ms" << std::endl;

	double diff = max_diff(mat2, mat3, M, N);
	std::cout << "diff between sequential and omp matrices: " << diff << std::endl;

	print_matrix_big(mat2, M, N, 3);
	print_matrix_big(mat3, M, N, 3);
}