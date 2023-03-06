// 2 Kernel Function for Array Power and ADC Energy

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <bitset>
#include <cuda_runtime.h>

// Copied from array_power.cu
#define INPUT_SIZE 256
#define OUTPUT_SIZE 256
#define FILTER_SIZE 3
#define CELL_PER_WEIGHT 4

// Copied from adc_power.cu
#define BIT 8
#define LENGTH 256
#define FILTER_NUM 32
#define ADC_EX 32 // (128 / 4)
#define TILE_SIZE_ARRAY 8
#define BLOCK_SIZE_ARRAY (TILE_SIZE_ARRAY+FILTER_SIZE-1)
#define TILE_SIZE 8
#define BLOCK_SIZE (TILE_SIZE+FILTER_SIZE-1)

#define CHECK_CUDA_ERROR(err) check((err), __FILE__, __LINE__)

/* Structure defination of conductance weight
 * An 8-bit weight to 4 devices for +MSB, +LSB, -MSB, -LSB
 */
typedef struct condSet{
    float P_MSB;
    float P_LSB;
    float N_MSB;
    float N_LSB;
} CondSet;

// Error Check
inline void check(cudaError_t err, const char* file, const int line);
// ADC Reference Array
float* adcRef(int n);
// Conductance Matrix
CondSet condWeight(float weight, float maxCond, float minCond);
// Array Power Kernel Function
__global__ void powerArray(int* input, float* weight_vec, float* power);
// ADC Energy Kernel Function
__global__ void energyADC(int* input, float* weight, float* lut, float* energy);

// Main Function
int main() 
{
    cudaError_t err = cudaSuccess;

    // Allocate Host Memory for Weight Conductance (ADC) and Conductance Summation (Array)
    /* Allocate Host Memory for Weight Conductance*/
    size_t size_condWeight = 3 * FILTER_SIZE * FILTER_SIZE * FILTER_NUM * CELL_PER_WEIGHT;
    float* h_condWeight = (float* ) malloc(size_condWeight * sizeof(float));
    if (h_condWeight == NULL) {
        fprintf(stderr, "Failed to allocate host memory at line %d!\n", __LINE__);
        exit(EXIT_FAILURE);
    }

    /* Allocate Host Memory for Weight Vector*/
    size_t size_condWeightVec = 3 * FILTER_SIZE * FILTER_SIZE;
    float* h_condWeightVec = (float* ) malloc(size_condWeightVec * sizeof(float));
    if (h_condWeightVec == NULL) {
        fprintf(stderr, "Failed to allocate host memory at line %d!\n", __LINE__);
        exit(EXIT_FAILURE);
    }

    // Load weight, covert weight to conductance
    float* tempLoadWeight = (float* ) malloc(3 * FILTER_SIZE * FILTER_SIZE * FILTER_NUM * sizeof(float));
    if (tempLoadWeight == NULL) {
        fprintf(stderr, "Failed to allocate host memory at line %d!\n", __LINE__);
        exit(EXIT_FAILURE);
    }

    FILE* pretrainedWeight;
    pretrainedWeight = fopen("weight.dat", "r");
    for (int i = 0; i < 3 * FILTER_SIZE * FILTER_SIZE * FILTER_NUM; ++i) {
        // weight in range [-1, 1]
        fscanf(pretrainedWeight, "%f", &tempLoadWeight[i]);
    }
    fclose(pretrainedWeight);

    // Convert Weight to Conducatance Array and Conductance Vector
    /* Conductance Array for ADC Energy Kernel Function */
    float maxCond = 100.0;
    float minCond = 1.0;
    for (int i = 0; i < 3 * FILTER_SIZE * FILTER_SIZE; ++i) { // Row
        for (int j = 0; j < FILTER_NUM; ++j) { // Column
            CondSet weights = condWeight(tempLoadWeight[FILTER_NUM*i+j], maxCond, minCond);
            
            h_condWeight[FILTER_NUM * CELL_PER_WEIGHT * i + CELL_PER_WEIGHT * j] = weights.P_MSB;
            h_condWeight[FILTER_NUM * CELL_PER_WEIGHT * i + CELL_PER_WEIGHT * j + 1] = weights.P_LSB;
            h_condWeight[FILTER_NUM * CELL_PER_WEIGHT * i + CELL_PER_WEIGHT * j + 2] = weights.N_MSB;
            h_condWeight[FILTER_NUM * CELL_PER_WEIGHT * i + CELL_PER_WEIGHT * j + 3] = weights.N_LSB;
        }
    }

    // Save Conductance Weight
    FILE* weight;
    char test[20] = "weightCond.dat";
    weight = fopen(test, "w");
    for (int i = 0; i < 27 * 128; ++i) {
        fprintf(weight, "%f\n", h_condWeight[i]);
    }

    fclose(weight);

    /* Conductance Vectro for Array Power Kernel Function*/
    for (int i = 0; i < 3 * FILTER_SIZE * FILTER_SIZE; ++i) {
        float p_sum = 0.0;
        for (int j = 0; j < FILTER_NUM * CELL_PER_WEIGHT; ++j) {
            p_sum += h_condWeight[i+j*3*FILTER_SIZE*FILTER_SIZE];
        }
        h_condWeightVec[i] = p_sum;
    }

    // Generate ADC LUT
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
    
    // Allocate Host Memory for Input, Overwrite this memory
    unsigned long size_input = 3 * INPUT_SIZE * INPUT_SIZE * BIT;
    int* h_input = (int* ) malloc(size_input * sizeof(int));
    if (h_input == NULL) {
        fprintf(stderr, "Failed to allocate host memory at line %d!\n", __LINE__);
		exit(EXIT_FAILURE);
    }

    // Allocate Host Memory for Array Power Output and ADC Energy Output
    unsigned long size_outputArray = OUTPUT_SIZE * OUTPUT_SIZE * BIT;
    float* h_outputArray = (float* ) malloc(size_outputArray * sizeof(float));
    if (h_outputArray == NULL) {
        fprintf(stderr, "Failed to allocate host memory at line %d!\n", __LINE__);
		exit(EXIT_FAILURE);
    }

    size_t size_outputADC = OUTPUT_SIZE * OUTPUT_SIZE * BIT * ADC_EX;
    float* h_outputADC = (float* ) malloc(size_outputADC * sizeof(float));
    if (h_outputADC == NULL) {
        fprintf(stderr, "Failed to allocate host memory at line %d!\n", __LINE__);
		exit(EXIT_FAILURE);
    }

    // Allocate Device Memory for 2 Conductance, ADC LUT, Input and 2 Outputs
    /* Allocate Memory for Conducatance Array*/
    float* d_condWeight = NULL;
    err = cudaMalloc((void**)&d_condWeight, size_condWeight * sizeof(float));
    CHECK_CUDA_ERROR(err);

    /* Allocate Memory for Conducatance Vector*/
    float* d_condWeightVec = NULL;
    err = cudaMalloc((void**)&d_condWeightVec, size_condWeightVec * sizeof(float));
    CHECK_CUDA_ERROR(err);

    /* Allocate Memory for ADC LUT*/
    float* d_adcRefArray = NULL;
    err = cudaMalloc((void**)&d_adcRefArray, size_lut * sizeof(float));
    CHECK_CUDA_ERROR(err);

    /* Allocate Memory for Input*/
    int* d_input = NULL;
    err = cudaMalloc((void**)&d_input, size_input * sizeof(int));
    CHECK_CUDA_ERROR(err);

    /* Allocate Memory for Array Power Output*/
    float* d_outputArray = NULL;
    err = cudaMalloc((void**)&d_outputArray, size_outputArray * sizeof(float));
    CHECK_CUDA_ERROR(err);

    /* Allocate Memory for ADC Energy Output*/
    float* d_outputADC = NULL;
    err = cudaMalloc((void**)&d_outputADC, size_outputADC * sizeof(float));
    CHECK_CUDA_ERROR(err);

    // Copy conductance matrix and ADC LUT to Device
    err = cudaMemcpy(d_condWeight, h_condWeight, size_condWeight * sizeof(float), cudaMemcpyHostToDevice);
    CHECK_CUDA_ERROR(err);

    err = cudaMemcpy(d_condWeightVec, h_condWeightVec, size_condWeightVec * sizeof(float), cudaMemcpyHostToDevice);
    CHECK_CUDA_ERROR(err);

    err = cudaMemcpy(d_adcRefArray, h_adcRefArray, size_lut * sizeof(float), cudaMemcpyHostToDevice);
    CHECK_CUDA_ERROR(err);

    // Allocate Host Memory for Fainal Output
    unsigned long size_finalOutput = OUTPUT_SIZE * OUTPUT_SIZE * (1 + ADC_EX);
    float* finalOutput = (float* ) malloc(size_finalOutput * sizeof(float));
    if (finalOutput == NULL) {
        fprintf(stderr, "Failed to allocate host memory at line %d!\n", __LINE__);
		exit(EXIT_FAILURE);
    }

    // Define Blocks, Threads and Streams
    /*Array Power*/
    dim3 arrayThreadsPerBlock(BLOCK_SIZE_ARRAY, BLOCK_SIZE_ARRAY, BIT);
    dim3 arrayBlocksPerGrid((INPUT_SIZE-1)/TILE_SIZE_ARRAY+1, (INPUT_SIZE-1)/TILE_SIZE_ARRAY+1, 1);
    /*ADC Energy*/
    dim3 ADCThreadsPerBlock(BLOCK_SIZE, BLOCK_SIZE, BIT);
    dim3 ADCBlocksPerGrid((INPUT_SIZE-1)/TILE_SIZE+1, (INPUT_SIZE-1)/TILE_SIZE+1, 1);
    
    /*2 Streams*/
    cudaStream_t stream1, stream2;
    err = cudaStreamCreate(&stream1);
    CHECK_CUDA_ERROR(err);
    err = cudaStreamCreate(&stream2);
    CHECK_CUDA_ERROR(err);
    
    /*Events*/
    cudaEvent_t event1, event2;
    err = cudaEventCreate(&event1);
    CHECK_CUDA_ERROR(err);
    err = cudaEventCreate(&event2);
    CHECK_CUDA_ERROR(err);

    // Loop through all input data
    size_t data_len = 2;
    for (int i = 0; i < data_len; ++i) {
        // Load input and copy input to Device
        FILE *myFile;
        char filename[100] = "../02_DataPreProcessing/01_tiff2bit/bit_dataRGB/data";
        sprintf(filename, "%s%d%s", filename, i, ".dat");
        myFile = fopen(filename, "r");
        for (int j = 0; j < size_input; ++j) {
            //h_input[j] = 1.0;
            fscanf(myFile, "%d", &h_input[j]);
        }
        fclose(myFile);
        
        err = cudaMemcpy(d_input, h_input, size_input * sizeof(int), cudaMemcpyHostToDevice);
        CHECK_CUDA_ERROR(err);

        // Run Kernel Functions
        powerArray<<<arrayBlocksPerGrid, arrayThreadsPerBlock, 0, stream1>>>(d_input, d_condWeightVec, d_outputArray);
        cudaEventRecord(event1, stream1);
        energyADC<<<ADCBlocksPerGrid, ADCThreadsPerBlock, 0, stream2>>>(d_input ,d_condWeight, d_adcRefArray, d_outputADC);
        cudaEventRecord(event2, stream2);

        err = cudaGetLastError();
        CHECK_CUDA_ERROR(err);
   
        cudaEventSynchronize(event1);
        cudaEventSynchronize(event2);

        // Allocate results and write to file
        err = cudaMemcpy(h_outputArray, d_outputArray, size_outputArray * sizeof(float), cudaMemcpyDeviceToHost);
        CHECK_CUDA_ERROR(err);
        err = cudaMemcpy(h_outputADC, d_outputADC, size_outputADC * sizeof(float), cudaMemcpyDeviceToHost);
        CHECK_CUDA_ERROR(err);

        // Data Pre-processing - Weighted Sum
        memset(finalOutput, 0.0, size_finalOutput * sizeof(float));
        for (unsigned long j = 0; j < BIT; ++j) {
            for (unsigned long k = 0; k < OUTPUT_SIZE; ++k) {
                for (unsigned long l = 0; l < OUTPUT_SIZE; ++l) {
                    finalOutput[k*OUTPUT_SIZE+l] += h_outputArray[OUTPUT_SIZE*BIT*k+BIT*l+j] * (1 << j);
                    for (unsigned long m = 0; m < ADC_EX; ++m) {
                        finalOutput[OUTPUT_SIZE*OUTPUT_SIZE+OUTPUT_SIZE*OUTPUT_SIZE*m+OUTPUT_SIZE*k+l] += 
                        h_outputADC[OUTPUT_SIZE*BIT*ADC_EX*k+BIT*ADC_EX*l+ADC_EX*j+m] * (1<<j);
                    }
                }
                //for (unsigned long l = 0; l < ADC_EX; ++l) {
                //    finalOutput[OUTPUT_SIZE*OUTPUT_SIZE + k*ADC_EX + l] +=
                //    h_outputADC[j*ADC_EX*OUTPUT_SIZE*OUTPUT_SIZE+k*ADC_EX+l] * (1 << j);
                //}
            }
        }
        
        FILE *myFile2;
        char filename2[80] = "/scratch/wluee_root/wluee1/ziwa/power";
        sprintf(filename2, "%s%d%s", filename2, i, ".dat");
        myFile2 = fopen(filename2, "w");

        for (unsigned long j = 0; j < size_finalOutput; ++j) {
            fprintf(myFile2, "%f\n", finalOutput[j]);
        }

        fclose(myFile2);
        
        // Below is original data saving without weigted sum
        /*
        // Write result to file
        FILE *myFile2;
        char filename2[20] = "power";
        sprintf(filename2, "%s%d%s", filename2, i, ".dat");
        myFile2 = fopen(filename2, "w");

        
        for (unsigned long j = 0; j < BIT; ++j) {
            for (unsigned long k = 0; k < OUTPUT_SIZE*OUTPUT_SIZE; ++k) {
                fprintf(myFile2, "%f\n", h_outputArray[j*OUTPUT_SIZE*OUTPUT_SIZE+k]);
                //for (unsigned long l = 0; l < ADC_EX; ++l) {
                //    fprintf(myFile2, "%f\n", h_outputADC[j*ADC_EX*OUTPUT_SIZE*OUTPUT_SIZE+k*ADC_EX+l]);
                //}
            }
        }
        
        fclose(myFile2);
        */
    }

    // Free Device Memory
    err = cudaFree(d_condWeight);
    CHECK_CUDA_ERROR(err);
    err = cudaFree(d_condWeightVec);
    CHECK_CUDA_ERROR(err);
    err = cudaFree(d_adcRefArray);
    CHECK_CUDA_ERROR(err);
    err = cudaFree(d_input);
    CHECK_CUDA_ERROR(err);
    err = cudaFree(d_outputArray);
    CHECK_CUDA_ERROR(err);
    err = cudaFree(d_outputADC);
    CHECK_CUDA_ERROR(err);

    // Free Host Memory
    free(h_condWeight);
    free(h_condWeightVec);
    free(h_adcRefArray);
    free(h_input);
    free(h_outputArray);
    free(h_outputADC);

    return 0;
}


inline void check(cudaError_t err, const char* file, const int line)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed in file %s, at line %d. Error code: %s\n",
                file, line, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}


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


CondSet condWeight(float weight, float maxCond = 100.0, float minCond = 1.0)
{
    CondSet condWeights;
    weight = weight * 255;
    //printf("Normalized Weight: %f\n", weight);

    if (weight >= 0) {
        condWeights.N_MSB = minCond;
        condWeights.N_LSB = minCond;
        std::bitset<BIT> quantWeight = (int) weight;
        condWeights.P_MSB = (maxCond - minCond) / 15 * 
                           ((quantWeight[7] << 3) + (quantWeight[6] << 2) + 
                            (quantWeight[5] << 1) + quantWeight[4]) + minCond;
        condWeights.P_LSB = (maxCond - minCond) / 15 * 
                           ((quantWeight[3] << 3) + (quantWeight[2] << 2) + 
                            (quantWeight[1] << 1) + quantWeight[0]) + minCond;
        //printf("Positive Weight - MSB: %f, LSB: %f\n", condWeights.P_MSB, condWeights.P_LSB);
        //printf("Bit Serial: ");
        //for (int i = 0; i < quantWeight.size(); ++i) {
        //    printf("%d", (int)quantWeight[7-i]);
        //}
        //printf("\n");
    }
    else {
        condWeights.P_MSB = minCond;
        condWeights.P_LSB = minCond;
        std::bitset<BIT> quantWeight = (int) -weight;
        condWeights.N_MSB = (maxCond - minCond) / 15 * 
                           ((quantWeight[7] << 3) + (quantWeight[6] << 2) + 
                            (quantWeight[5] << 1) + quantWeight[4]) + minCond;
        condWeights.N_LSB = (maxCond - minCond) / 15 * 
                           ((quantWeight[3] << 3) + (quantWeight[2] << 2) + 
                            (quantWeight[1] << 1) + quantWeight[0]) + minCond;
        //printf("Negative Weight - MSB: %f, LSB: %f\n", condWeights.N_MSB, condWeights.N_LSB);
        //printf("Bit Serial: ");
        //for (int i = 0; i < quantWeight.size(); ++i) {
        //    printf("%d", (int)quantWeight[7-i]);
        //}
        //printf("\n");
    }

    return condWeights;
}


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
    
    int row_o = blockIdx.y * TILE_SIZE_ARRAY + ty;
    int col_o = blockIdx.x * TILE_SIZE_ARRAY + tx;
    int dep_o = tz;

    int row_i = row_o - FILTER_SIZE / 2;
    int col_i = col_o - FILTER_SIZE / 2;

    // Load the Input Tile to shared memory
    __shared__ int s_input[BLOCK_SIZE_ARRAY][BLOCK_SIZE_ARRAY][3*BIT];
    if ((row_i >= 0 && row_i < INPUT_SIZE) && 
        (col_i >= 0 && col_i < INPUT_SIZE)) {
        for (int depth = 0; depth < 3*BIT; ++depth) {
            s_input[ty][tx][depth] = input[row_i*INPUT_SIZE*BIT*3 + 
                                           col_i*BIT*3 + depth];
        }
    } else {
        for (int depth = 0; depth < 3*BIT; ++depth) {
            s_input[ty][tx][depth] = 0.0f;
        }
    }

    __syncthreads();
    // threads for row scan, column scan, bit-serial scan
    // a for loop for vector dot product
    float output = 0.0f;
    if (tx < TILE_SIZE_ARRAY && ty < TILE_SIZE_ARRAY && tz < BIT) {
        for (int i = 0; i < FILTER_SIZE; ++i) {
            for (int j = 0; j < FILTER_SIZE; ++j) {
                for (int k = 0; k < 3; ++k) {
                    output += s_weight[i*FILTER_SIZE*FILTER_SIZE + j*FILTER_SIZE + k] 
                            * s_input[i+ty][j+tx][k*BIT+tz];
                }
            }
        }
        if (row_o < OUTPUT_SIZE && col_o < OUTPUT_SIZE && dep_o < BIT) {
            power[row_o*OUTPUT_SIZE*BIT + col_o*BIT + dep_o] = output;
        }
    }
}

__global__
void energyADC(int* input, float* weight, float* lut, float* energy)
{
    // Load ADC energy LUT to shared memory
    __shared__ float s_lut[LENGTH];
    for (int i = 0; i < LENGTH; ++i) {
        s_lut[i] = lut[i];
    }

    // Load all conv kernels to shared memory
    //__shared__ float s_weight[3 * FILTER_SIZE * FILTER_SIZE * FILTER_NUM * CELL_PER_WEIGHT];
    //for (int i = 0; i < 3*FILTER_SIZE*FILTER_SIZE*FILTER_NUM*CELL_PER_WEIGHT; ++i) {
    //    s_weight[i] = weight[i];
    //}
    __shared__ float s_weight[3*FILTER_SIZE*FILTER_SIZE][FILTER_NUM*CELL_PER_WEIGHT];
    for (int i = 0; i < 3*FILTER_SIZE*FILTER_SIZE; ++i) {
        for (int j = 0; j < FILTER_NUM*CELL_PER_WEIGHT; ++j) {
            s_weight[i][j] = weight[i*FILTER_NUM*CELL_PER_WEIGHT + j];
        }
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
    __shared__ int s_input[BLOCK_SIZE][BLOCK_SIZE][3*BIT];
    if ((row_i >= 0 && row_i < INPUT_SIZE) && 
        (col_i >= 0 && col_i < INPUT_SIZE)) {
        for (int depth = 0; depth < 3*BIT; ++depth) {
            s_input[ty][tx][depth] = input[row_i*INPUT_SIZE*BIT*3 + 
                                           col_i*BIT*3 + depth];
        }
    } else {
        for (int depth = 0; depth < 3*BIT; ++depth) {
            s_input[ty][tx][depth] = 0.0f;
        }
    }

    __syncthreads();

    // threads for row scan, column scan, bit-serial scan
    // a for loop for vector dot product
    float result[ADC_EX]; // An array stores ADC energy of each output feature
    if (tx < TILE_SIZE && ty < TILE_SIZE && tz < BIT) {
        for (int k_idx = 0; k_idx < ADC_EX; ++k_idx) {
            float adc_energy = 0.0f; // total ADC energy of each ADC execution cycle
            for (int col = 0; col < 4; ++col) {
                float output1 = 0.0f; // partial result of the first tile
                for (int i = 0; i < FILTER_SIZE; ++i) {
                    for (int j = 0; j < FILTER_SIZE; ++j) {
                        for (int k = 0; k < 3; ++k) {
                            output1 += s_weight[i*FILTER_SIZE*FILTER_SIZE + j*FILTER_SIZE + k][k_idx*4 + col]
                                    * s_input[i+ty][j+tx][k*BIT+tz];
                        }
                    }
                }
                // Result done, indexing ADC energy here
                // Scale to [0,255]
                int index1 = output1 > 255.0 ? 255 : (int) output1; // Scale the output range
                adc_energy += s_lut[index1];
            }
            // Total energy of all ADC works together
            result[k_idx] = adc_energy;
        }
        if (row_o < OUTPUT_SIZE && col_o < OUTPUT_SIZE && dep_o < BIT) {
            for (int i = 0; i < ADC_EX; ++i) {
                energy[(row_o*OUTPUT_SIZE*BIT + col_o*BIT + dep_o) * ADC_EX + i] = result[i];
            }
        }
    }
}

