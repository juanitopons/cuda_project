#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <curand.h>
#include <curand_kernel.h>
#include <time.h>
#include <cuda_profiler_api.h>
#include <sys/time.h>

const int REP_PAR = 32; // Iterations for a given parallel execution size
const int REP_SEQ = 2; // Iterations for a given sequentiall execution size
const int LENGHT = 2; // Lenght of the random sequence numbers
const int N_MAX = 4194304; // End size
const int N = 1024; // Initial size
const int THREAD_MAX = 1024; // Max threads per block


////////////////////////////
// GPU Parallel algorithm //
////////////////////////////

/**
 * [oddEvenSortEven description] GPU function. Even execution of Odd-Even Mergesort algorithm.
 *
 * @param dSequence  [description] Device (GPU) array to be sorted-
 * @param height     [description] Actual sub-butterfly (actual height).
 * @param actualDeep [description] Actual sub-butterfly iteration deep (up to down).
 * @param k          [description] Size of the array to be sorted
 * @param threads    [description] Threads must have the actual kernel call (we can't create unpair total number of thread in a pair number of blocks)
 * @param netHeight  [description] Bitonic global height
 */
__global__ void oddEvenSortEven(int *dSequence, int height, int actualDeep, int k, int threads, int netHeight){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(x < threads) {
        int indexA;
        int indexB;
        int offset = (1 << (actualDeep - 1));
        int relation = ((1 << height) / 2) - offset;
        int numThread = x % relation;
        int mult = x / relation;

        if(x < actualDeep) {
            indexA = offset + x;
        } else {
            indexA = (numThread / offset) * offset + numThread + (mult * (1 << height)) + offset;
        }

        indexB = indexA + (1 << (actualDeep - 1));

        int aux = 0;

        if(dSequence[indexA] > dSequence[indexB]) {
            aux = dSequence[indexA];
            dSequence[indexA] = dSequence[indexB];
            dSequence[indexB] = aux;
        }
    }

}

/**
 * [oddEvenSortOdd description] GPU function. Odd execution of Odd-Even Mergesort algorithm.
 *
 * @param dSequence  [description] Device (GPU) array to be sorted-
 * @param height     [description] Actual sub-butterfly (actual height).
 * @param actualDeep [description] Actual sub-butterfly iteration deep (up to down).
 * @param k          [description] Size of the array to be sorted
 */
__global__ void oddEvenSortOdd(int *dSequence, int height, int actualDeep, int k){
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if(x < (k / 2)) {
        int indexA;
        int indexB;
        int offset = (1 << (actualDeep - 1));
        
        if(x < (offset - 1)) {
            indexA = x;
            indexB = indexA + offset;
        } else {
            indexA = (x / offset) * offset + x;
            indexB = indexA + offset;
        }

        int aux = 0;

        if(dSequence[indexA] > dSequence[indexB]) {
            aux = dSequence[indexA];
            dSequence[indexA] = dSequence[indexB];
            dSequence[indexB] = aux;
        }
    }
}


///////////////////////////////
// CPU Sequentiall Algorithm //
///////////////////////////////

/**
 * [oddEvenSortEven description] Even execution of Odd-Even Mergesort algorithm.
 *
 * @param dSequence  [description] Array to be sorted.
 * @param height     [description] 
 * @param actualDeep [description]
 * @param iteration  [description]
 * @param k          [description] Size of the array.
 */
void oddEvenSortEven(int *dSequence, int height, int actualDeep, int iteration, int k){
    int indexA;
    int indexB;
    int offset = (1 << (actualDeep - 1));
    int relation = ((1 << height) / 2) - offset;
    int numThread = iteration % relation;
    int mult = iteration / relation;

    if(iteration < actualDeep) {
        indexA = offset + iteration;
    } else {
        indexA = (numThread / offset) * offset + numThread + (mult * (1 << height)) + offset;
    }

    indexB = indexA + (1 << (actualDeep - 1));

    int aux = 0;

    if(dSequence[indexA] > dSequence[indexB]) {
        aux = dSequence[indexA];
        dSequence[indexA] = dSequence[indexB];
        dSequence[indexB] = aux;
    }
    
}

/**
 * [oddEvenSortOdd description]
 *
 * @param dSequence  [description] Array to be sorted.
 * @param height     [description]
 * @param actualDeep [description]
 * @param iteration  [description] 
 * @param k          [description] Size of the array.
 */
void oddEvenSortOdd(int *dSequence, int height, int actualDeep, int iteration, int k){
    int indexA;
    int indexB;
    int offset = (1 << (actualDeep - 1));
    
    if(iteration < (offset - 1)) {
        indexA = iteration;
        indexB = indexA + offset;
    } else {
        indexA = (iteration / offset) * offset + iteration;
        indexB = indexA + offset;
    }

    int aux = 0;

    if(dSequence[indexA] > dSequence[indexB]) {
        aux = dSequence[indexA];
        dSequence[indexA] = dSequence[indexB];
        dSequence[indexB] = aux;
    }
}

/**
 * [oddEvenSortSec description] It does the sequential execution of the Odd-Even mergesort
 * algorithm for a given array of ints.
 *
 * @param dSequence [description] Array to be sorted
 * @param n         [description] Size of the array
 */
void oddEvenSortSec(int *dSequence, int n){
    int height = log((double)n) / log(2.0);
    for(int i = 1; i <= height; i++) {
        for(int j = i; j > 0; j--) {
            int iterations;
            if(j == i) {
                iterations = n / 2;
                for(int k = 0; k < iterations; k++) {
                    oddEvenSortOdd(dSequence, i, j, k, n);
                }
            } else {
                iterations = (n / 2) - ((1 << (j - 1)) * (n / (1 << i)));
                for(int k = 0; k < iterations; k++) {
                    oddEvenSortEven(dSequence, i, j, k, n);
                }
            }
        }
    }
}


/////////////////////
// Usage Functions //
/////////////////////

/**
 * [initArray description] Initialize, randomly, a sequence of numbers in an array of ints.
 *
 * @param h_array [description] Array to allocate the sequence.
 * @param k       [description] Size of the sequence.
 */
void initArray(int *h_array, int k) {
    int mult = pow(10.0, LENGHT);
    for (int i = 0; i < k; i++){
        h_array[i] = rand() % mult;
    }
}

/**
 * [checkArrayGPU description] GPU function. Evaluates and checks the correct execution of the
 * algorithm after it finish.
 *
 * @param d_array [description] Device (GPU) array to be printed.
 * @param k       [description] Size of the array.
 */
__global__ void checkArrayGPU(int *d_array, int k){
    int boolean = 1;
    for(int i = 0; i < k - 1; i++) {
        if(d_array[i] > d_array[i + 1]) {
            boolean = 0;
            break;
        }
    }
    if(d_array[k - 1] == 0)
        boolean = 0;

    printf("Boolean = %d; (if 1 all OK)\n", boolean);
}

/**
 * [printArrayCPU description] Prints the array from the CPU.
 *
 * @param h_array [description] Array to be printed.
 * @param k       [description] Size of the array.
 */
void printArrayCPU(int * h_array, int k) {
    for(int i = 0; i < k; i++) {
        printf("- %d -", h_array[i]);
    }
    printf("\n");
}

/**
 * [printArrayGPU description] GPU function. Prints the array from the GPU.
 *
 * @param d_array [description] Device (GPU) array to be printed.
 * @param k       [description] Size of the array.
 */
__global__ void printArrayGPU(int * d_array, int k) {
    for(int i = 0; i < k; i++) {
        printf("- %d -", d_array[i]);
    }
    printf("\n");
}

/**
 * [checkArrayCPU description] Evaluates and checks the correct execution of the algorithm after
 * it finish.
 *
 * @param h_array [description] Device (GPU) array to be printed.
 * @param k       [description] Size of the array.
 */
void checkArrayCPU(int *h_array, int k){
    int boolean = 1;
    for(int i = 0; i < k - 1; i++) {
        if(h_array[i] > h_array[i + 1]) {
            boolean = 0;
            break;
        }
    }
    if(h_array[k - 1] == 0)
        boolean = 0;

    printf("Boolean = %d; (if 1 all OK)\n", boolean);
}


//////////
// MAIN //
//////////

int main( int argc, char* argv[] )
{
    // File for saving data times
    FILE *f1 = fopen("odd_even_merge_sort_parallel.dat", "w");
    fprintf(f1, "N\tBlocks\tThreads/Block\tT_Par\t\tT_Seq\t\tSpeed-Up\n");

    // Time structures variables for sequentiall and parallel calculation
    struct timeval t1, t2, t1_seq, t2_seq;
    double elapsedTime = 0, elapsedTimeSec = 0;

    for(int k = N; k < N_MAX; k*=2) {

        // Total threads we must have for each iteration of the algorithm
        int threads;

        // Threads per each block in the GPU
        int threadsPerBlock;

        // Total number of blocks in the GPU
        int numBlocks;

        // Bitonic height
        int height = log(k) / log(2);
        
        // Elements data arrays (CPU and (h_) GPU (d_))
        int h_array[k];
        int* d_array;

        // Random array initialization
        initArray(h_array, k);

        // Array memory allocation
        cudaMalloc(&d_array, k * sizeof(int));


        //////////////
        // PARALELL //
        //////////////

        for(int m = 0; m < REP_PAR; m++) {
            cudaMemcpy(d_array, h_array, k * sizeof(int), cudaMemcpyHostToDevice);
            gettimeofday(&t1, 0); // Started time
            cudaProfilerStart(); // Profiler start
            for(int i = 1; i <= height; i++) {
                for(int j = i; j > 0; j--) {
                    if(j == i) {
                        threadsPerBlock = k / 2;
                        if(threadsPerBlock > THREAD_MAX) {
                            threadsPerBlock = THREAD_MAX;
                        }
                        numBlocks = ceil(((double)k/2) / threadsPerBlock);
                        oddEvenSortOdd<<<numBlocks,threadsPerBlock>>>(d_array, i, j, k);
                    } else {
                        threads = (k / 2) - (pow(2.0, (double)(j - 1)) * (k / pow(2.0, (double)i)));
                        if(threads > THREAD_MAX) {
                            threadsPerBlock = THREAD_MAX;
                        }
                        numBlocks = ceil(((double)threads / THREAD_MAX));
                        oddEvenSortEven<<<numBlocks,threadsPerBlock>>>(d_array, i, j, k, threads, height);
                    }
                }
            }
            cudaThreadSynchronize(); // Synchronization
            cudaProfilerStop(); // Profiler stop
            gettimeofday(&t2, 0); // Stopped time
            elapsedTime += (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000000.0;
        }
    
        printf("N = %d - > Ended Paralell\n", k);
        checkArrayGPU<<<1,1>>>(d_array, k);

        
        /////////////////
        // Sequentiall //
        /////////////////

        // Array copy auxiliar
        int h_array2[k];

        for(int m = 0; m < REP_SEQ; m++) {
            memcpy(h_array2, h_array, k * sizeof(int));
            gettimeofday(&t1_seq, 0);
            //cudaProfilerStart();
            oddEvenSortSec(h_array2, k);
            //cudaProfilerStop();
            gettimeofday(&t2_seq, 0);
            elapsedTimeSec += (1000000.0*(t2_seq.tv_sec-t1_seq.tv_sec) + t2_seq.tv_usec-t1_seq.tv_usec)/1000000.0;
        }
        printf("N = %d - > Ended Sequentiall\n", k);
        checkArrayCPU(h_array2, k);

        
        ///////////////////////////////////
        // Times and SpeedUp calculation //
        ///////////////////////////////////

        elapsedTime = elapsedTime / REP_PAR;
        elapsedTimeSec = elapsedTimeSec / REP_SEQ;
        float speedup = elapsedTimeSec / elapsedTime;

        fprintf(f1,"%d\t%d\t%d\t\t%.6fs\t%.6fs\t%.3f\n",k, numBlocks, threadsPerBlock, elapsedTime, elapsedTimeSec, speedup);

        cudaFree(d_array); // Free GPU memory

    }

    cudaDeviceSynchronize();

    fclose(f1); // Close file

    return 0;
}