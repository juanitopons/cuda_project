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

const int LENGHT = 2;
const int N_MAX = 2097152;
const int N = 4;
//const int threadsPerBlock = 32;

__global__ void initArrayParallel(int *dSequence, int *dSequenceSeq, curandState *state, unsigned long seed, int n){

    int x = blockDim.x * blockIdx.x + threadIdx.x;
    curand_init(seed, x, 0, &state[x]);
    dSequence[x] = curand_uniform(&state[x]) * pow(10.0, (double)LENGHT);
    dSequenceSeq[x] = dSequence[x];
    dSequence[n-x-1] = curand_uniform(&state[x]) * pow(10.0, (double)LENGHT);
    dSequenceSeq[n-x-1] = dSequence[n-x-1];

}

__device__ void bitonicMergeMax(int *dSequence, int actualDeep){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = (int)pow(2.0, (double) actualDeep) / 2;
    int x = index + (index& ~(offset - 1));
    int j = index + (index& ~(offset - 1)) + offset;
    int aux = 0;

    if(dSequence[x] < dSequence[j]) {
        aux = dSequence[x];
        dSequence[x] = dSequence[j];
        dSequence[j] = aux;
    }
}

__device__ void bitonicMergeMin(int *dSequence, int actualDeep){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = (int)pow(2.0, (double) actualDeep) / 2;
    int x = index + (index& ~(offset - 1));
    int j = index + (index& ~(offset - 1)) + offset;
    int aux = 0;

    if(dSequence[x] > dSequence[j]) {
        aux = dSequence[x];
        dSequence[x] = dSequence[j];
        dSequence[j] = aux;
    }
}

__global__ void bitonicSort(int *dSequence, int height, int actualDeep){

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int newHeight = height - 1;
    newHeight = pow(2.0, (double)newHeight);
    if((x&newHeight) == 0) {
        bitonicMergeMin(dSequence, actualDeep);
    } else {
        bitonicMergeMax(dSequence, actualDeep);
    }

}

void bitonicMergeMaxSec(int *dSequence, int k, int actualDeep){
    int offset = (int)pow(2.0, (double) actualDeep) / 2;
    int x = k + (k& ~(offset - 1));
    int j = k + (k& ~(offset - 1)) + offset;
    int aux = 0;

    if(dSequence[x] < dSequence[j]) {
        aux = dSequence[x];
        dSequence[x] = dSequence[j];
        dSequence[j] = aux;
    }
}

void bitonicMergeMinSec(int *dSequence, int k, int actualDeep){
    int offset = (int)pow(2.0, (double) actualDeep) / 2;
    int x = k + (k& ~(offset - 1));
    int j = k + (k& ~(offset - 1)) + offset;
    int aux = 0;

    if(dSequence[x] > dSequence[j]) {
        aux = dSequence[x];
        dSequence[x] = dSequence[j];
        dSequence[j] = aux;
    }
}

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

__global__ void printArray(int *d_array, int k){
    int boolean = 1;
    for(int i = 0; i < k - 1; i++) {
        if(d_array[i] > d_array[i + 1]) {
            boolean = 0;
            break;
        }
    }
    if(array[k - 1] == 0)
        boolean = 0;

    printf("Boolean = %d; (if 1 all OK)\n\n", boolean);
}


int main( int argc, char* argv[] )
{
    FILE *f1 = fopen("bitonic_sort_1048576.dat", "w");
    fprintf(f1, "N\tBlocks\tThreads/Block\tT_Par\t\tT_Seq\t\tSpeed-Up\n");

    struct timeval t1, t2, t1_seq, t2_seq;
    double elapsedTime, elapsedTimeSec;

    int inc = 2;
    for(int k = N; k < N_MAX; k*=2) {
        int* dSequence;
        int* dSequenceSeq;
        int hSequenceSeq[k];
        int numThreads = k / 2;
        int threadsPerBlock = log(inc) / log(2);
        if(threadsPerBlock > 1) threadsPerBlock = threadsPerBlock / 2;
        threadsPerBlock = pow(2.0, (double) threadsPerBlock);
        int numBlocks = numThreads / threadsPerBlock;
        int height = log(k) / log(2);
        curandState *d_state;

        cudaMalloc(&dSequence, k * sizeof(int));
        cudaMalloc(&dSequenceSeq, k * sizeof(int));
        cudaMalloc(&d_state, numThreads * sizeof(curandState));

        initArrayParallel<<<numBlocks,threadsPerBlock>>>(dSequence, dSequenceSeq, d_state, time(NULL), k);
        cudaFree(d_state);
        /** Parallel **/

        cudaProfilerStart();
        gettimeofday(&t1, 0);
        for(int i = 1; i <= height; i++) {
            for(int j = i; j > 0; j--) {
                bitonicSort<<<numBlocks,threadsPerBlock>>>(dSequence, i, j);
            }
        }
        cudaThreadSynchronize();
        gettimeofday(&t2, 0);
        cudaProfilerStop();
        printf("N = %d - > Acabado Paralelo\n", k);
        printArray<<<1,1>>>(dSequence, k);

        /** END Parallel **/

        /** Sequentiall **/
        cudaMemcpy(hSequenceSeq, dSequenceSeq, k * sizeof(int), cudaMemcpyDeviceToHost);

        cudaProfilerStart();
        gettimeofday(&t1_seq, 0);
        bitonicSortSec(hSequenceSeq, k);
        gettimeofday(&t2_seq, 0);
        cudaProfilerStop();
        printf("N = %d - > Acabado Secuencial\n", k);
        printArray<<<1,1>>>(hSequenceSeq, k);

        /** END Sequentiall **/

        elapsedTime = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000000.0;
        elapsedTimeSec = (1000000.0*(t2_seq.tv_sec-t1_seq.tv_sec) + t2_seq.tv_usec-t1_seq.tv_usec)/1000000.0;
        float speedup = elapsedTimeSec / elapsedTime;

        fprintf(f1,"%d\t%d\t%d\t\t%.6fs\t%.6fs\t%.3f\n",k, numBlocks, threadsPerBlock, elapsedTime, elapsedTimeSec, speedup);

        //printArray<<<1,1>>>(dSequence);
        inc *= 2;

        cudaFree(dSequence);
        cudaFree(dSequenceSeq);

        cudaDeviceSynchronize();
    }

    fclose(f1);

    return 0;
}