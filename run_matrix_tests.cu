/*goal: compare performance of a few sequential & parallel implementations of matrix multiplication*/

#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <omp.h>
#include <chrono>
#include <random>
#include <iomanip>

int random_uniform_initialize(double* m, int M, int N, double lower, double upper)
{
	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution(lower, upper);

	for (int i = 0; i < M * N; ++i)
	{
		*(m + i) = distribution(generator);
	}
	return 0;
}

int sequential_naive_multiply(double* A, double* B, double* C, int M, int K, int N)
{
	for (int i = 0; i < M * N; ++i)
	{
		double val = 0.;
		int row = i % M;
		int col = i / M;
		for (int j = 0; j < K; ++j)
		{
			val += *(A + j * M + row) * *(B + col * K + j);
		}
		*(C + i) = val;
	}
	return 0;
}

void print_matrix(const double* mat, int M, int N)
{
	//std::cout << std::setw(8);
	std::cout << std::endl;
	for (int i = 0; i < M; ++i)
	{
		for (int j = 0; j < N; ++j)
		{
			std::cout << std::fixed << std::setprecision(3) << std::setw(8) << *(mat + j * M + i) << " ";
		}
		std::cout << std::endl;
	}
}

int main()
{
	int M = 8;
	int N = 8;
	double mat1[64];
	int init_failed = random_uniform_initialize(mat1, M, N, 0.0, 10.5);
	print_matrix(mat1, 8, 8);


	std::cout << "test sequential naive multiply:" << std::endl;
	double A[4] = { 2., 0., 0., 3. };
	double B[4] = { 2., 0., 1., 2. };
	double C[4];
	sequential_naive_multiply(A, B, C, 2, 2, 2);
	print_matrix(C, 2, 2);
}