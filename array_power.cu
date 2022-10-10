/// Copyright (c) 2022 Ziyu Wang @ University of Michigan, Ann Arbor
/// All rights reserved
/// 
/// Contributors:
///   - Ziyu Wang (ziwa[AT]umich[DOT]edu)
///

#include <stdio.h>

#define BIT_PRECISION 8

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
   - int filterNum: number of Conv filter in one layer (may not necessary)
   - float* weight: pre-trained NN weight after reshape & reprecision 
                    to fit RRAM array
                    length - [2 * BIT_PRECISION/DEVICE_PRECISION * filterNumber * kernelSize]
                    For Conv layer, store this in the shared memory
   - float* power : Power consumption of each slide step in convolution
                    For Mo x Mo output feature map size, length is Mo x Mo x BIT_PRECISION
**/

__global__
void powerArray(int length, int filterNum, int* input, float* weight, float* power)
{
    // Row-wise summation of weight, simplify the problem to vector dot product
    // load vectored weight shared memory

    // threads for row scan, column scan, bit-serial scan
    // a for loop for vector dot product
}

int main()
{
    // Allocate host memory for input data
   
    // Load input data to host memory

    // Allocate host memory for conv filter

    // Load Conv filter
    // Use reandom floating number at this moment

    // Allocate host memory for array power

    // Allocate device memory for input data, conv filter and array power

    // Copy data from host to device

    // Define kernel and run kernel function

    // Copy data from device to host

    // Result check

    // Write result to file
    // To Do: set the file as a PyTorch trainable dataset

    // free memory
    return 0;
}
