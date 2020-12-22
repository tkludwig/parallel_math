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
	const int M = 99;
	const int N = 99;
	const int K = 99;
	double mat1[M * K];
	double mat2[K * N];
	double mat3[M * N]; //array to store result from sequential multiply
	double mat4[M * N]; //array to store result from omp multiply
	double mat5[M * N]; //host array to store result from gpu multiply

	double* d_mat1; //device pointer
	double* d_mat2;
	double* d_mat3;

	cudaMalloc(&d_mat1, M * N * sizeof(double));
	cudaMalloc(&d_mat2, M * N * sizeof(double));
	cudaMalloc(&d_mat3, M * N * sizeof(double));

	random_uniform_initialize(mat1, M, N, 0.0, 0.5);
	random_uniform_initialize(mat2, M, N, 0.0, 0.5);

	std::cout << "mat1 initial:" << std::endl;
	print_matrix_big(mat1, M, N, 3);

	std::cout << "test sequential naive multiply:" << std::endl;
	auto start = std::chrono::steady_clock::now();
	naive_multiply_sequential(mat1, mat2, mat3, M, K, N);
	auto end = std::chrono::steady_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	std::cout << "naive_multiple_sequential on size " << M << " took " << duration.count() << " ms" << std::endl;

	std::cout << "test omp naive multiply:" << std::endl;
	start = std::chrono::steady_clock::now();
	naive_multiply_omp(mat1, mat2, mat4, M, K, N);
	end = std::chrono::steady_clock::now();
	duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	std::cout << "naive_multiple_omp on size " << M << " took " << duration.count() << " ms" << std::endl;

	std::cout << "test cuda naive multiply:" << std::endl;
	//ordinarily I would wrap the cudaMemcpy into the host function, but I want to see actually how long these mem copies take
	auto start_kernel = std::chrono::steady_clock::now();
	auto end_kernel = std::chrono::steady_clock::now();

	start = std::chrono::steady_clock::now();


	cudaMemcpy(d_mat1, mat1, M * K * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_mat2, mat2, K * N * sizeof(double), cudaMemcpyHostToDevice);

	start_kernel = std::chrono::steady_clock::now();
	naive_multiply_cuda(d_mat1, d_mat2, d_mat3, M, K, N);
	end_kernel = std::chrono::steady_clock::now();
	cudaMemcpy(mat5, d_mat3, M * N * sizeof(double), cudaMemcpyDeviceToHost);

	end = std::chrono::steady_clock::now();
	duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	auto kernel_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_kernel - start_kernel);

	std::cout << "naive_multiple_cuda on size " << M << " took " << duration.count() << " ms" << std::endl;
	std::cout << "naive_multiple_cuda kernel on size " << M << " took " << kernel_duration.count() << " ms" << std::endl;

	print_matrix_big(mat3, M, N, 3);
	print_matrix_big(mat4, M, N, 3);
	print_matrix_big(mat5, M, N, 3);
}