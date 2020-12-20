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
}