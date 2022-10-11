/// Copyright (c) 2022 Ziyu Wang @ University of Michigan, Ann Arbor
/// All rights reserved
/// 
/// Contributors:
///   - Ziyu Wang (ziwa[AT]umich[DOT]edu)
///

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define BIT_PRECISION 8
#define INPUT_SIZE 256
#define OUTPUT_SIZE 256
#define FILTER_SIZE 3
#define FILTER_NUM 16
#define CELL_PER_WEIGHT 4
#define TILE_SIZE 8
#define BLOCK_SIZE (TILE_SIZE+FILTER_SIZE-1)

#define CHECK_CUDA_ERROR(err) check((err), __FILE__, __LINE__)
inline void check(cudaError_t err, const char* file, const int line)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed in file %s, at line %d. Error code: %s\n",
                file, line, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

/* Kernel Function for RRAM Crossbar Power at Runtime
 * The weight, input may map onto multiple tiles,
 * but the power of each tile can be allocated together.
 * This function calculate the total power of a NN layer
 * inference at runtime for each input bit.
 * 
 * Input:
 * - int length: the bit serial input. [N, N, C] input image
 *               has a length of N x N x C x BIT_PRCISION
 * - int* input: A pointer to the input bit-serial array
                 Image arranges as row-major in the array
                 3D[N, N, C] -> 1D[R,G,B,R,G,B,...] ->
                 1D[R(LSB,...MSB),G(LSB,...,MSB),...]
                 The l-th LSB of the row i, col j, channel k
                 is input[i * (N * C * BIT_PRECISION) + 
                 j * (C * BIT_PRECISION) + C * BIT_PRECISION + l]
                 3D format [N, N, C * BIT_PRECISION]
   - int filterNum: number of Conv filter in one layer (may not necessary)
   - float* weight_vec: pre-trained NN weight after reshape & reprecision 
                        to fit RRAM array
                        length - [3 * FILTER_SIZE^2 * FILTER_NUM * CELL_PER_WEIGHT]
                        vectorized to 1D [3 * FILTER_SIZE^2] for array power computation
                        Store this in the shared memory
   - float* power : Power consumption of each slide step in convolution
                    For Mo x Mo output feature map size, length is Mo x Mo x BIT_PRECISION
**/

__global__
void powerArray(int* input, float* weight_vec, float* power)
{
    // Load vectored weight shared memory
    __shared__ float s_weight[3 * FILTER_SIZE * FILTER_SIZE];
    for (int i = 0; i < 3*FILTER_SIZE*FILTER_SIZE; ++i) {
        s_weight[i] = weight_vec[i];
    }
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    
    int row_o = blockIdx.y * TILE_SIZE + ty;
    int col_o = blockIdx.x * TILE_SIZE + tx;
    int dep_o = tz;

    int row_i = row_o - FILTER_SIZE / 2;
    int col_i = col_o - FILTER_SIZE / 2;

    // Load the Input Tile to shared memory
    __shared__ int s_input[BLOCK_SIZE][BLOCK_SIZE][3*BIT_PRECISION];
    if ((row_i >= 0 && row_i < INPUT_SIZE) && 
        (col_i >= 0 && col_i < INPUT_SIZE)) {
        for (int depth = 0; depth < 3*BIT_PRECISION; ++depth) {
            s_input[ty][tx][depth] = input[row_i*INPUT_SIZE*BIT_PRECISION*3 + 
                                           col_i*BIT_PRECISION*3 + depth];
        }
    } else {
        for (int depth = 0; depth < 3*BIT_PRECISION; ++depth) {
            s_input[ty][tx][depth] = 0.0f;
        }
    }

    __syncthreads();
    // threads for row scan, column scan, bit-serial scan
    // a for loop for vector dot product
    float output = 0.0f;
    if (tx < TILE_SIZE && ty < TILE_SIZE && tz < BIT_PRECISION) {
        for (int i = 0; i < FILTER_SIZE; ++i) {
            for (int j = 0; j < FILTER_SIZE; ++j) {
                for (int k = 0; k < 3; ++k) {
                    output += s_weight[i*FILTER_SIZE*FILTER_SIZE + k*FILTER_SIZE + k] 
                            * s_input[i+ty][j+tx][k+tz];
                }
            }
        }
        if (row_o < OUTPUT_SIZE && col_o < OUTPUT_SIZE && dep_o < BIT_PRECISION) {
            power[row_o*OUTPUT_SIZE*BIT_PRECISION + col_o*BIT_PRECISION + dep_o] = output;
        }
    }
}

int main()
{
    cudaError_t err = cudaSuccess;

    // Allocate host memory for input data
    size_t size_input = 3 * INPUT_SIZE * INPUT_SIZE * BIT_PRECISION;
    int* h_input = (int* ) malloc(size_input * sizeof(int));
    if (h_input == NULL) {
        fprintf(stderr, "Failed to allocate host memory at line %d!\n", __LINE__);
		exit(EXIT_FAILURE);
    }
    
    // Load input data to host memory
    // To Do: change the code to scan the whole folder
    FILE *myFile;
    myFile = fopen("test.dat", "r");
    for (int i = 0; i < size_input; ++i) {
        fscanf(myFile, "%d", &h_input[i]);
    }
    // Allocate host memory for conv filter
    size_t size_filter = 3 * FILTER_SIZE * FILTER_SIZE * FILTER_NUM * CELL_PER_WEIGHT;
    float* h_filter = (float* ) malloc(size_filter * sizeof(float));
    if (h_filter == NULL) {
        fprintf(stderr, "Failed to allocate host memory at line %d!\n", __LINE__);
		exit(EXIT_FAILURE);
    }

    // Load Conv filter
    // Use reandom floating number at this moment
    for (int i = 0; i < size_filter; ++i) {
        h_filter[i] = (float)rand() / RAND_MAX;
    }
  
    // Vectorize(summation) filter in x-axis for array power computation
    // Light workload and only done once in the host
    size_t size_filter_vec = 3 * FILTER_SIZE * FILTER_SIZE;
    float* h_filter_vec = (float* ) malloc(size_filter_vec * sizeof(float));
    if (h_filter_vec == NULL) {
        fprintf(stderr, "Failed to allocate host memory at line %d!\n", __LINE__);
		exit(EXIT_FAILURE);
    }
    
    for (int i = 0; i < 3 * FILTER_SIZE * FILTER_SIZE; ++i) {
        float p_sum = 0.0;
        for (int j = 0; j < FILTER_NUM * CELL_PER_WEIGHT; ++j) {
            p_sum += h_filter[i * FILTER_NUM * CELL_PER_WEIGHT + j];
        }
        h_filter_vec[i] = p_sum;
    }

    // Allocate host memory for array power
    size_t size_output = OUTPUT_SIZE * OUTPUT_SIZE * BIT_PRECISION;
    float* h_output = (float* ) malloc(size_output * sizeof(float));
    if (h_output == NULL) {
        fprintf(stderr, "Failed to allocate host memory at line %d!\n", __LINE__);
		exit(EXIT_FAILURE);
    }

    // Allocate device memory for input data, conv filter and array power
    int* d_input = NULL;
    err = cudaMalloc((void** )&d_input, size_input * sizeof(int));
    CHECK_CUDA_ERROR(err);
    float* d_filter_vec = NULL;
    err = cudaMalloc((void** )&d_filter_vec, size_filter_vec * sizeof(float));
    CHECK_CUDA_ERROR(err);
    float* d_output = NULL;
    err = cudaMalloc((void** )&d_output, size_output * sizeof(float));
    CHECK_CUDA_ERROR(err);

    // Copy data from host to device
    err = cudaMemcpy(d_input, h_input, size_input * sizeof(int), cudaMemcpyHostToDevice);
    CHECK_CUDA_ERROR(err);
    err = cudaMemcpy(d_filter_vec, h_filter_vec, size_filter_vec * sizeof(float), cudaMemcpyHostToDevice);
    CHECK_CUDA_ERROR(err);

    // Define kernel and run kernel function
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE, BIT_PRECISION);
    dim3 blocksPerGrid((INPUT_SIZE-1)/TILE_SIZE+1, (INPUT_SIZE-1)/TILE_SIZE+1, 1);
 
    powerArray<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_filter_vec, d_output);
    err = cudaGetLastError();
    CHECK_CUDA_ERROR(err);

    // Copy data from device to host
    err = cudaMemcpy(h_output, d_output, size_output * sizeof(float), cudaMemcpyDeviceToHost);
    CHECK_CUDA_ERROR(err);

    // Result check

    // Write result to file
    // To Do: set the file as a PyTorch trainable dataset

    // free device memory
    err = cudaFree(d_input);
    CHECK_CUDA_ERROR(err);
    err = cudaFree(d_filter_vec);
    CHECK_CUDA_ERROR(err);
    err = cudaFree(d_output);
    CHECK_CUDA_ERROR(err);

    // free host memory
    free(h_input);
    free(h_filter);
    free(h_filter_vec);
    free(h_output);

    return 0;
}
