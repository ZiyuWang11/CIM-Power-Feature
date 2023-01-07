/// Copyright (c) 2022 Ziyu Wang @ University of Michigan, Ann Arbor
/// All rights reserved
/// 
/// Contributors:
///   - Ziyu Wang (ziwa[AT]umich[DOT]edu)
///

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define BIT 8
#define LENGTH 256
#define ADC_EX 32 // (128 / 4)
#define BIT_PRECISION 8
#define INPUT_SIZE 256
#define OUTPUT_SIZE 256
#define FILTER_SIZE 3
#define FILTER_NUM 64
#define CELL_PER_WEIGHT 4
#define TILE_SIZE 4
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

/* Function to generate ADC execution energy LUT
 * Only compute the DAC swithing energy at this moment, as it contributed most
 * Option - Add comparator and register switching energy
 * The function returns an array of swithing energy with different output code
 * Array length is 256 for 8-bit SAR ADC, functon can be generalized
 * The array will be loaded to device memory for VMM results
 * VMM results need to scale to the int index range, [0, 255]
**/

float* adcRef(int bit = 8)
{
    // Create an array to be returned
    static float arr[LENGTH];

    // Initialize cap array
    int c[BIT+1];
    int sumc = 0;
    for (int i = 0; i < BIT; ++i) {
        c[i] = 1 << (BIT - i - 1);
        sumc += c[i];
    }
    // Ref cap
    c[BIT+1] = 1;
    sumc += 1;
   
    float vref = 1.0;

    // Loop through all output codes
    // i is normalized Vin
    for (int i = 0; i < LENGTH; ++i) {
        float vin = (float)i / (float)(LENGTH - 1);
        // Initialize switches, comparator voltage and energy
        int s[BIT];
        memset(s, 0, sizeof(s[0])*BIT);

        float vcomp = 0.0;
        float vcomp_prev = 0.0;
        float energy = 0.0;

        // Initial charge
        float q = - vin * sumc;
        
        // Loop through all DAC switching
        for (int j = 0; j < BIT; ++j) {
            // close switch
            s[j] = 1;
            // generate a new vcomp
            vcomp_prev = vcomp;
            float c_connect = 0.0;
            for (int k = 0; k < BIT; ++k) {
                c_connect += s[k] * c[k];
            }
            vcomp = (q + c_connect * vref) / sumc;

            if (j == 0) {
                energy += c[j] * vref * (- vin - (vcomp - vref));
            }
            else {
                float c_switch = 0.0;
                for (int k = 0; k < j; ++k) {
                    c_switch += s[k] * c[k];
                }
                energy += c_switch * vref * (vcomp_prev - vcomp);
                energy += c[j] * vref * (vcomp_prev - (vcomp - vref));
            }

            s[j] = vcomp < 0.0 ? 1 : 0;
        }
       
        arr[i] = energy;
    }

    return arr;
}

/* Kernel Function for ADC execution function at runtime
 * Only consider the first layer of U-Net, 64 kernels mapped on 2 tiles
 * Each array is arranged with 4 ADCs, and 8 ADC energy indexing a time
 * Every 4 columns partial results if for a weight
 * This function calculate the total power of a NN layer
 * inference at runtime for each output AD conversion.
 * 
 * Input:
 * - int* input: A pointer to the input bit-serial array
                 Image arranges as row-major in the array
                 3D[N, N, C] -> 1D[R,G,B,R,G,B,...] ->
                 1D[R(LSB,...MSB),G(LSB,...,MSB),...]
                 The l-th LSB of the row i, col j, channel k
                 is input[i * (N * C * BIT_PRECISION) + 
                 j * (C * BIT_PRECISION) + C * BIT_PRECISION + l]
                 3D format [N, N, C * BIT_PRECISION]
   - float* weight_vec: pre-trained NN weight after reshape & reprecision 
                        to fit RRAM array
                        length - [3 * FILTER_SIZE^2 * FILTER_NUM * CELL_PER_WEIGHT]
                        Store this in the shared memory
   - float* energy_adc: ADC total energy LUT, calculated in host and used as 
                        a LUT in device. 256 length for 8-bit SAR ADC
   - float* energy : Switching energy of each A/D data conversion
                     For Mo x Mo output feature map size, length is Mo x Mo x (#col/#ADC) x BIT_PRECISION
**/

__global__
void powerArray(int* input, float* weight, float* lut, float* energy)
{
    // Load ADC energy LUT to shared memory
    __shared__ float s_lut[LENGTH];
    for (int i = 0; i < LENGTH; ++i) {
        s_lut[i] = lut[i];
    }

    // Load all conv kernels to shared memory
    __shared__ float s_weight[3 * FILTER_SIZE * FILTER_SIZE * FILTER_NUM * CELL_PER_WEIGHT];
    for (int i = 0; i < 3*FILTER_SIZE*FILTER_SIZE*FILTER_NUM*CELL_PER_WEIGHT; ++i) {
        s_weight[i] = weight[i];
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
    float result[ADC_EX]; // An array stores ADC energy of each output feature
    if (tx < TILE_SIZE && ty < TILE_SIZE && tz < BIT_PRECISION) {
        for (int k_idx = 0; k_idx < ADC_EX; ++k_idx) {
            float adc_energy = 0.0f; // total ADC energy of each ADC execution cycle
            for (int col = 0; col < 4; ++col) {
                float output1 = 0.0f; // partial result of the first tile
                float output2 = 0.0f; // partial result of the second tile
                for (int i = 0; i < FILTER_SIZE; ++i) {
                    for (int j = 0; j < FILTER_SIZE; ++j) {
                        for (int k = 0; k < 3; ++k) {
                            output1 += s_weight[k_idx*4*27 + col*27 + i*FILTER_SIZE*FILTER_SIZE + j*FILTER_SIZE + k] 
                                    * s_input[i+ty][j+tx][k+tz];
                            output2 += s_weight[3456 + k_idx*4*27 + col*27 + i*FILTER_SIZE*FILTER_SIZE + j*FILTER_SIZE + k] 
                                    * s_input[i+ty][j+tx][k+tz];
                        }
                    }
                }
                // Result done, indexing ADC energy here
                // Scale to [0,255]
                int index1 = output1 > 255.0 ? 255 : (int)output1;
                int index2 = output2 > 255.0 ? 255 : (int)output2;
                adc_energy += s_lut[index1] + s_lut[index2];
            }
            // Total energy of all ADC works together
            result[k_idx] = adc_energy;
        }
        if (row_o < OUTPUT_SIZE && col_o < OUTPUT_SIZE && dep_o < BIT_PRECISION) {
            for (int i = 0; i < ADC_EX; ++i) {
                energy[(row_o*OUTPUT_SIZE*BIT_PRECISION + col_o*BIT_PRECISION + dep_o) * ADC_EX + i] = result[i];
            }
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
  
    // Allocate host memory for ADC enerygy
    size_t size_output = OUTPUT_SIZE * OUTPUT_SIZE * BIT_PRECISION * ADC_EX;
    float* h_output = (float* ) malloc(size_output * sizeof(float));
    if (h_output == NULL) {
        fprintf(stderr, "Failed to allocate host memory at line %d!\n", __LINE__);
		exit(EXIT_FAILURE);
    }

    // Get ADC energy LUT
    size_t size_lut = LENGTH;
    float* h_adcRefArray = (float* ) malloc(size_lut * sizeof(float));
    if (h_adcRefArray == NULL) {
        fprintf(stderr, "Failed to allocate host memory at line %d!\n", __LINE__);
		exit(EXIT_FAILURE);
    }
    float* test_adcRefArray = adcRef(BIT);
    for (int i = 0; i < LENGTH; ++i) {
        h_adcRefArray[i] = test_adcRefArray[i];
    }

    // Allocate device memory for input data, conv filter and ADC energy
    int* d_input = NULL;
    err = cudaMalloc((void** )&d_input, size_input * sizeof(int));
    CHECK_CUDA_ERROR(err);
    float* d_filter = NULL;
    err = cudaMalloc((void** )&d_filter, size_filter * sizeof(float));
    CHECK_CUDA_ERROR(err);
    float* d_adcRefArray = NULL;
    err = cudaMalloc((void** )&d_adcRefArray, size_lut * sizeof(float));
    CHECK_CUDA_ERROR(err);
    float* d_output = NULL;
    err = cudaMalloc((void** )&d_output, size_output * sizeof(float));
    CHECK_CUDA_ERROR(err);

    // Copy data from host to device
    err = cudaMemcpy(d_input, h_input, size_input * sizeof(int), cudaMemcpyHostToDevice);
    CHECK_CUDA_ERROR(err);
    err = cudaMemcpy(d_filter, h_filter, size_filter * sizeof(float), cudaMemcpyHostToDevice);
    CHECK_CUDA_ERROR(err);
    err = cudaMemcpy(d_adcRefArray, h_adcRefArray, size_lut * sizeof(float), cudaMemcpyHostToDevice);
    CHECK_CUDA_ERROR(err);

    // Define kernel and run kernel function
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE, BIT_PRECISION);
    dim3 blocksPerGrid((INPUT_SIZE-1)/TILE_SIZE+1, (INPUT_SIZE-1)/TILE_SIZE+1, 1);
 
    powerArray<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_filter, d_adcRefArray, d_output);
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
    err = cudaFree(d_filter);
    CHECK_CUDA_ERROR(err);
    err = cudaFree(d_adcRefArray);
    CHECK_CUDA_ERROR(err);
    err = cudaFree(d_output);
    CHECK_CUDA_ERROR(err);

    // free host memory
    free(h_input);
    free(h_filter);
    free(h_adcRefArray);
    free(h_output);

    return 0;
}
