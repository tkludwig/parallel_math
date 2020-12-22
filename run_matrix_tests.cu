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

	std::cout << "Armadillo version??: " << arma::arma_version::as_string() << std::endl;
	const int M = 9;
	const int N = 9;
	double mat1[M * N];
	double mat2[M * N];
	double mat3[M * N];
	double mat4[M * N];
	random_uniform_initialize(mat1, M, N, 0.0, 0.5);
	random_uniform_initialize(mat2, M, N, 0.0, 0.5);

	arma::Mat<double> a_mat1(mat1, M, N, true);
	arma::Mat<double> a_mat2(mat2, M, N, true);
	arma::Mat<double> a_mat3(M, N);

	std::cout << "mat1 initial:" << std::endl;
	print_matrix_big(mat1, M, N, 3);
	std::cout << "a_mat1 initial:" << std::endl;
	print_matrix_big(a_mat1.memptr(), M, N, 3);

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

	a_mat3 = a_mat1 * a_mat2; //the libraries required for this are not linking

	double diff = max_diff(mat3, mat4, M, N);
	std::cout << "diff between sequential and omp matrices: " << diff << std::endl;
	diff = max_diff(mat3, a_mat3.memptr(), M, N);
	std::cout << "diff between omp and armadillo results: " << diff << std::endl;


	print_matrix_big(mat3, M, N, 3);
	print_matrix_big(mat4, M, N, 3);
	print_matrix_big(a_mat3.memptr(), M, N, 3);
}