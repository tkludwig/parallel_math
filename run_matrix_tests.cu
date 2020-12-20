/*goal: compare performance of a few sequential & parallel implementations of matrix multiplication*/

#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <omp.h>
#include <chrono>