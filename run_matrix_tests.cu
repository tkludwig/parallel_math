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
	const int M = 9;
	const int N = 9;
	double mat1[M * N];
	double mat2[M * N];
	double mat3[M * N];
	double mat4[M * N];
	random_uniform_initialize(mat1, M, N, 0.0, 0.5);
	random_uniform_initialize(mat2, M, N, 0.0, 0.5);

	std::cout << "mat1 initial:" << std::endl;
	print_matrix_big(mat1, M, N, 3);

	std::cout << "test omp naive multiply:" << std::endl;
	auto start = std::chrono::high_resolution_clock::now();
	naive_multiply_omp(mat1, mat2, mat3, M, N, N);
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	std::cout << "naive_multiple_omp on size " << M << " took " << duration.count() << " ms" << std::endl;

	start = std::chrono::high_resolution_clock::now();
	naive_multiply_sequential(mat1, mat2, mat4, M, N, N);
	end = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	std::cout << "naive_multiple_sequential on size " << M << " took " << duration.count() << " ms" << std::endl;

	print_matrix_big(mat3, M, N, 3);
	print_matrix_big(mat4, M, N, 3);
}