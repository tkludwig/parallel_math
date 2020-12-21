/*goal: compare performance of a few sequential & parallel implementations of matrix multiplication*/

#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <omp.h>
#include <chrono>
#include <random>

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

int main()
{
	int M = 50;
	int N = 60;
	double mat1[3000];
	std::cout << "initial, final values of mat1: " << mat1[0] << " " << mat1[1] << " " <<
		mat1[2] << " ... " << mat1[2997] << " " << mat1[2998] << " " << mat1[2999] << std::endl;
	int init_failed = random_uniform_initialize(mat1, M, N, 0.0, 0.5);
	std::cout << "initial, final values of mat1: " << mat1[0] << " " << mat1[1] << " " <<
		mat1[2] << " ... " << mat1[2997] << " " << mat1[2998] << " " << mat1[2999] << std::endl;

	std::cout << "test sequential naive multiply:" << std::endl;
	double A[4] = { 2., 0., 0., 3. };
	double B[4] = { 2., 0., 1., 2. };
	double C[4];
	sequential_naive_multiply(A, B, C, 2, 2, 2);
	std::cout << "C: " << C[0] << " " << C[1] << " " << C[2] << " " << C[3] << std::endl;
}