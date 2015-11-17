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
 * [bitonicMergeMax description] GPU function. Does the MAX exchange of the bitonic algorithm.
 *
 * @param dSequence  [description] Array to be exchanged.
 * @param index      [description] Thread unique ID.
 * @param offset     [description] Actual iteration sub-butterfly deep (up to down).
 */
__device__ void bitonicMergeMax(int *dSequence, int index, int offset){
    int x;
    int j;

    if(index < (offset - 1)) {
        x = index;
        j = x + offset;

    } else {
        x = (index / offset) * offset + index;
        j = x + offset;
    }

    int aux = 0;

    if(dSequence[x] < dSequence[j]) {
        aux = dSequence[x];
        dSequence[x] = dSequence[j];
        dSequence[j] = aux;
    }
}

/**
 * [bitonicMergeMin description] GPU function. Does the MIN exchange of the bitonic algorithm.
 *
 * @param dSequence  [description] Array to be exchanged.
 * @param index      [description] Thread unique ID.
 * @param offset     [description] Actual iteration sub-butterfly deep (up to down) offset.
 */
__device__ void bitonicMergeMin(int *dSequence, int index, int offset){
    int x;
    int j;

    if(index < (offset - 1)) {
        x = index;
        j = x + offset;

    } else {
        x = (index / offset) * offset + index;
        j = x + offset;
    }

    int aux = 0;

    if(dSequence[x] > dSequence[j]) {
        aux = dSequence[x];
        dSequence[x] = dSequence[j];
        dSequence[j] = aux;
    }


}

/**
 * [bitonicSort description] GPU funciton. Executes the Bitonic Sort hard-coded algorithm in the
 * GPU.
 *
 * @param dSequence  [description] Device (GPU) array to be sorted.
 * @param height     [description] Actual sub-butterfly (actual height).
 * @param actualDeep [description] Actual iteration sub-butterfly deep (up to down).
 * @param k          [description] Size of dSequence.
 */
__global__ void bitonicSort(int *dSequence, int height, int actualDeep, int k){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if(x < (k / 2)) {
        int offset = (1 << (actualDeep - 1));
        height = (1 << (height - 1));

        if((x & height) == 0) {
            bitonicMergeMin(dSequence, x, offset);
        } else {
            bitonicMergeMax(dSequence, x, offset);
        }
    }

}


///////////////////////////////
// CPU Sequentiall Algorithm //
///////////////////////////////

/**
 * [bitonicMergeMaxSec description] Does the MAX exchange of the bitonic algorithm.
 *
 * @param dSequence  [description] Array to be exchanged.
 * @param k          [description] Size of the array.
 * @param actualDeep [description] Actual iteration sub-butterfly deep.
 */
void bitonicMergeMaxSec(int *dSequence, int k, int actualDeep){
    int offset = (int)pow(2.0, (double) (actualDeep - 1));
    int x;
    int j;

    if(k < (offset - 1)) {
        x = k;
        j = x + offset;

    } else {
        x = (k / offset) * offset + k;
        j = x + offset;
    }

    int aux = 0;

    if(dSequence[x] < dSequence[j]) {
        aux = dSequence[x];
        dSequence[x] = dSequence[j];
        dSequence[j] = aux;
    }
}

/**
 * [bitonicMergeMinSec description]Does the MIN exchange of the bitonic algorithm.
 *
 * @param dSequence  [description] Array to be exchanged.
 * @param k          [description] Size of the array.
 * @param actualDeep [description] Actual iteration sub-butterfly deep.
 */
void bitonicMergeMinSec(int *dSequence, int k, int actualDeep){
    int offset = (int)pow(2.0, (double) (actualDeep - 1));
    int x;
    int j;

    if(k < (offset - 1)) {
        x = k;
        j = x + offset;

    } else {
        x = (k / offset) * offset + k;
        j = x + offset;
    }

    int aux = 0;

    if(dSequence[x] > dSequence[j]) {
        aux = dSequence[x];
        dSequence[x] = dSequence[j];
        dSequence[j] = aux;
    }
}

/**
 * [bitonicSortSec description] Executes the Bitonic sort algorithm sequentially in the CPU.
 *
 * @param dSequence [description] Array to be sorted.
 * @param n         [description] Size of the array.
 */
void bitonicSortSec(int *dSequence, int n){
    int height = log((double)n) / log(2.0);
    for(int i = 1; i <= height; i++) {
        for(int j = i; j > 0; j--) {
            for(int k = 0; k < (n / 2); k++) {
                 int newHeight = i - 1;
                 newHeight = pow(2.0, (double)newHeight);
                 if((k&newHeight) == 0) {
                     bitonicMergeMinSec(dSequence, k, j);
                 } else {
                     bitonicMergeMaxSec(dSequence, k, j);
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


//////////
// MAIN //
//////////

int main( int argc, char* argv[] )
{
    // File for saving data times
    FILE *f1 = fopen("bitonic_sort_parallel.dat", "w");
    fprintf(f1, "N\tBlocks\tThreads/Block\tT_Par\t\tT_Seq\t\tSpeed-Up\n");

    // Time structures variables for sequentiall and parallel calculation
    struct timeval t1, t2, t1_seq, t2_seq;
    double elapsedTime = 0, elapsedTimeSec = 0;

    for(int k = N; k < N_MAX; k*=2) {
        
        // Threads per each block in the GPU
        int threadsPerBlock = k / 2;
        if(threadsPerBlock > THREAD_MAX)
            threadsPerBlock = THREAD_MAX;
        
        // Total number of blocks in the GPU
        int numBlocks = ceil(((double)k/2) / threadsPerBlock);

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
                    bitonicSort<<<numBlocks,threadsPerBlock>>>(d_array, i, j, k);
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
            bitonicSortSec(h_array2, k);
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