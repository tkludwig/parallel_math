#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <omp.h>
#include <chrono>
#include <random>
#include <iomanip>
#include <cmath>

#include "matrix_operations.cuh"

#define BLOCKSIZE_1 16

__global__ void kernel_naive_multiply_cuda(double* Ad, double* Bd, double* Cd, int M, int K, int N)
{
	double val = 0.;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row >= 0 && row < M && col >= 0 && col < N)
	{
		for (int k = 0; k < K; ++k)
		{
			val += *(Ad + (row)+k * M) * *(Bd + (col * K) + k);
		}
	}
	*(Cd + row + col * M) = val;
}

int naive_multiply_cuda(double* __restrict__ A, double* __restrict__ B,
	double* __restrict__ C, int M, int K, int N)
{
	dim3 threads(BLOCKSIZE_1, BLOCKSIZE_1); //threads per block
	unsigned int blocks_x = (N + threads.x - 1) / threads.x; //basically ceil(N/threads.x)
	unsigned int blocks_y = (M + threads.y - 1) / threads.y;
	dim3 blocks(blocks_x, blocks_y);
	kernel_naive_multiply_cuda<<<blocks, threads>>>(A, B, C, M, K, N);
	return 0;
}

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

int naive_multiply_sequential(double* A, double* B, double* C, int M, int K, int N)
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

int naive_multiply_omp(double* A, double* B, double* C, int M, int K, int N)
{
#pragma omp parallel for
	for (int i = 0; i < M * N; ++i)
	{
		//printf("hello from thread %d", omp_get_thread_num());
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

void print_matrix_big(const double* mat, int M, int N, int include)
{
	std::cout << std::endl << "[" << std::endl;
	for (int i = 0; i < include; ++i)
	{
		for (int j = 0; j < include; ++j)
		{
			std::cout << std::fixed << std::setprecision(3) << std::setw(8) << *(mat + j * M + i) << " ";
		}
		std::cout << "        ..." << std::endl;
	}
	std::cout << "                ...";
	for (int i = (M - include); i < M; ++i)
	{
		std::cout << std::endl << "                ...    ";
		for (int j = (N - include); j < N; ++j)
		{
			std::cout << std::fixed << std::setprecision(3) << std::setw(8) << *(mat + j * M + i) << " ";
		}
	}
	std::cout << std::endl << "]" << std::endl;
}

double max_diff(const double* mat1, const double* mat2, int M, int N)
{
	double maxd = 0.;
	double tdiff = 0.;
	for (int i = 0; i < M * N; ++i)
	{
		tdiff = std::abs(*(mat1 + i) - *(mat2 + i));
		if (tdiff > maxd)
		{
			maxd = tdiff;
		}
	}
	return maxd;
}