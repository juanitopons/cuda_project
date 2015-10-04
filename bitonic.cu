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
const int N_MAX = 4194304;
const int N = 64;
const int THREAD_MAX = 1024;

void initArray(int *h_array, int k) {
    int mult = pow(10.0, LENGHT);
    for (int i = 0; i < k; i++){
        h_array[i] = rand() % mult;
    }
}

__device__ void bitonicMergeMax(int *dSequence, int actualDeep, int index){
    int offset = (1 << (actualDeep - 1));
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

__device__ void bitonicMergeMin(int *dSequence, int actualDeep, int index){
    int offset = (1 << (actualDeep - 1));
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

__global__ void bitonicSort(int *dSequence, int height, int actualDeep, int k){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if(x < (k / 2)) {
        int newHeight = height - 1;
        newHeight = (1 << newHeight);

        if((x & newHeight) == 0) {
            bitonicMergeMin(dSequence, actualDeep, x);
        } else {
            bitonicMergeMax(dSequence, actualDeep, x);
        }
    }

}

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

void printArrayCPU(int * h_array, int k) {
    for(int i = 0; i < k; i++) {
        printf("- %d -", h_array[i]);
    }
    printf("\n");
}

__global__ void printArrayGPU(int * d_array, int k) {
    for(int i = 0; i < k; i++) {
        printf("- %d -", d_array[i]);
    }
    printf("\n");
}

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


int main( int argc, char* argv[] )
{
    FILE *f1 = fopen("bitonic_sort_1048576.dat", "w");
    fprintf(f1, "N\tBlocks\tThreads/Block\tT_Par\t\tT_Seq\t\tSpeed-Up\n");

    struct timeval t1, t2, t1_seq, t2_seq;
    double elapsedTime, elapsedTimeSec;

    int inc = 2;
    for(int k = N; k < N_MAX; k*=2) {
        int threadsPerBlock = k / 2;
        if(threadsPerBlock > THREAD_MAX)
            threadsPerBlock = THREAD_MAX;
        int numBlocks = ceil(((double)k/2) / threadsPerBlock);
        int height = log(k) / log(2);
        
        int h_array[k];
        int* d_array;

        initArray(h_array, k);

        cudaMalloc(&d_array, k * sizeof(int));
        cudaMemcpy(d_array, h_array, k * sizeof(int), cudaMemcpyHostToDevice);

        /** PARALELL **/

        gettimeofday(&t1, 0);
        for(int i = 1; i <= height; i++) {
            for(int j = i; j > 0; j--) {
                bitonicSort<<<numBlocks,threadsPerBlock>>>(d_array, i, j, k);
            }
        }
        cudaThreadSynchronize();
        gettimeofday(&t2, 0);
        checkArrayGPU<<<1,1>>>(d_array, k);
        printf("N = %d - > Acabado Paralelo\n", k);
        //checkArrayGPU<<<1,1>>>(d_array, k);

        /** >>>>>>> **/


        /** Sequentiall **/
        
        cudaProfilerStart();
        gettimeofday(&t1_seq, 0);
        bitonicSortSec(h_array, k);
        gettimeofday(&t2_seq, 0);
        cudaProfilerStop();
        printf("N = %d - > Acabado Secuencial\n", k);

        /** >>>>>>> **/

        elapsedTime = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000000.0;
        elapsedTimeSec = (1000000.0*(t2_seq.tv_sec-t1_seq.tv_sec) + t2_seq.tv_usec-t1_seq.tv_usec)/1000000.0;
        float speedup = elapsedTimeSec / elapsedTime;

        fprintf(f1,"%d\t%d\t%d\t\t%.6fs\t%.6fs\t%.3f\n",k, numBlocks, threadsPerBlock, elapsedTime, elapsedTimeSec, speedup);

        inc *= 2;

        cudaFree(d_array);

    }

    cudaDeviceSynchronize();

    fclose(f1);

    return 0;
}