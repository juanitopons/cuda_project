#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <curand.h>
#include <curand_kernel.h>
#include <time.h>
#include <cuda_profiler_api.h>
#include <sys/time.h>

const int REP_PAR = 30;
const int REP_SEQ = 2;
const int LENGHT = 2;
const int N_MAX = 65536;
const int N = 16384;
const int THREAD_MAX = 1024;
int threadsPerBlock;
int numBlocks;

typedef unsigned int Branch;

struct ThreadT
{
	unsigned int blockIdx;
	unsigned int threadIdx;
};

struct CudaT
{
	ThreadT thread;
	unsigned int time;
};

struct Point
{
	unsigned int sbf;
	unsigned int row;
	unsigned int col;
};


int flip(int k, int n) {
    return n ^ (1 << k);
}

__device__ int d_flip(int k, int n) {
    return n ^ (1 << k);
}

int grow(int b, int r) {
	return b * (b - 1) / 2 + b - r;
}

__device__ int bit(int position, int x) {
	return (x >> position) & 1;
}

int h_bit(int x, int position) {
	return (x >> position) & 1;
}

void initArray(int *h_array, int k) {
    int mult = pow(10.0, LENGHT);
    for (int i = 0; i < k; i++){
        h_array[i] = rand() % mult;
    }
}

__global__ void printArrayGPU(int * d_array, int k) {
    for(int i = 0; i < k; i++) {
        //printf("- %d [%d] -", d_array[i], i);
        printf("- %d -", d_array[i]);
    }
    printf("\n");
}

__device__ void printArrayGPU2(int * d_array, int k) {
    for(int i = 0; i < k; i++) {
        printf("- %d [%d] -", d_array[i], i);
    }
    printf("\n");
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

// DDA

__device__ bool DI(Point p, int h) {
	return (0 < p.sbf <= h) &&
	((p.row < p.sbf) || (p.row <= p.sbf && p.sbf == 1)) &&
	(p.col < (1 << h));
}

Point sp(Point p, Branch b, int h) {
	if((p.sbf != h) && (p.row == 0)) {
		if(b == 0) return {p.sbf + 1, p.sbf, p.col};
		else return {p.sbf + 1, p.sbf, flip(p.sbf, p.col)};

	} else if(((p.sbf == 1) && (p.row == 1)) ||
		(p.sbf > 1) && (1 <= p.row <= (p.sbf - 1))) {
		if(b == 0) return {p.sbf, p.row - 1, p.col};
		else return {p.sbf, p.row - 1, flip(p.row - 1, p.col)};
	}
}

__device__ Point d_sp(Point p, Branch b, int h) {
	if((p.sbf != h) && (p.row == 0)) {
		if(b == 0) return {p.sbf + 1, p.sbf, p.col};
		else return {p.sbf + 1, p.sbf, d_flip(p.sbf, p.col)};

	} else if(((p.sbf == 1) && (p.row == 1)) ||
		(p.sbf > 1) && (1 <= p.row <= (p.sbf - 1))) {
		if(b == 0) return {p.sbf, p.row - 1, p.col};
		else return {p.sbf, p.row - 1, d_flip(p.row - 1, p.col)};
	}
}


bool rg(Point p, Branch b) {
	return (p.row != 1 || p.sbf != 1);
}


__device__ Point d_rp(Point p, Branch b) {
	if(((p.sbf > 1) && (0 <= p.row && p.row <= (p.sbf - 2))) ||
		((p.sbf == 1) && (p.row == 0))) {
		if(b == 0) return {p.sbf, p.row + 1, p.col};
		else return {p.sbf, p.row + 1, d_flip(p.row, p.col)};

	} else if((p.sbf > 1) && (p.row = p.sbf - 1)) {
		if(b == 0) return {p.sbf - 1, 0, p.col};
		else return {p.sbf - 1, 0, d_flip(p.row, p.col)};
	}
}

__device__ bool cond(Point p) {
	return (bit(p.sbf, p.col) == bit(p.row, p.col));
}

// CUDA Space Time Threads

bool DI(ThreadT t) {
	return ((t.blockIdx < numBlocks) && (t.threadIdx < THREAD_MAX));
}


CudaT in(CudaT t, ThreadT ch) {
	return {ch, t.time-1};
}

CudaT out(CudaT t, ThreadT ch) {
	return {ch, t.time+1};
}

CudaT em(Point p) { // Embed Point to our FFT
	ThreadT t = {(p.col / threadsPerBlock), (p.col % threadsPerBlock)};
	return {t, grow(p.sbf, p.row)};
}


__global__ void bitonicSortDDA(int * d_array, int start, int end, int rowStart, int rowEnd, int h, int k) {
	extern __shared__ int local_array[];
	int x = blockIdx.x * blockDim.x + threadIdx.x;

	if(x < k) {
		Point p = {start, rowStart, x};
		int num1 = d_array[x];
		int num2 = d_array[d_rp(p, 1).col];
		int range = rowStart;
		int stop = 0;

		for(int i = start; i <= end; i++) {
			if(i == end)
				stop = rowEnd;

			for(int j = range; j >= stop; j--) {
				if(bit(p.sbf, p.col) == bit(p.row, p.col)) {
					//MIN
					if(end == 9 && rowEnd == 8 && (x == 0 || x == 256)) {
						//printf("MIN: Comparando [%d] con [%d]\n", p.col, d_rp(p, 1).col);
					}
					num1 = min(num1, num2);
				} else {
					if(end == 9 && rowEnd == 8 && (x == 0 || x == 256)) {
						//printf("MAX: Comparando [%d] con [%d]\n", p.col, d_rp(p, 1).col);
					}
					//MAX
					num1 = max(num1, num2);
				}
				p = d_sp(p, 0, h);
				local_array[threadIdx.x] = num1;

				__syncthreads();

				num2 = local_array[d_rp(p, 1).col % blockDim.x];

				__syncthreads();

			}
			range = p.row;
			//__syncthreads();
	    }

	    d_array[x] = local_array[threadIdx.x];
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

int main( int argc, char* argv[] )
{
    FILE *f1 = fopen("bitonic_sort_dda.dat", "w");
    fprintf(f1, "N\tBlocks\tThreads/Block\tT_Par\t\tT_Seq\t\tSpeed-Up\n");

    struct timeval t1, t2, t1_seq, t2_seq;
    double elapsedTime = 0, elapsedTimeSec = 0;

    int inc = 2;
    for(int k = N; k < N_MAX; k*=2) {
		threadsPerBlock = THREAD_MAX;
		if(THREAD_MAX > k)
			threadsPerBlock = k;

    	numBlocks = k / threadsPerBlock;
		int h = log(k) / log(2);

		int h_array[k];
	    int* d_array;

	    int tmax = (h * (h + 1)) / 2;
	    int numPoints = 1 << tmax;

	    int SBF[tmax];
	    int ROW[tmax];
		int a = 0;
		int t = 0;

		initArray(h_array, k);

		cudaMalloc(&d_array, k * sizeof(int));

		int onlyWithinBlock = 1;
		int branch = 1;
		int range = 1;
		int sbf;
		int row;

		for(int i = 1; i <= h; i++) {
			for(int j = range; j >= 0; j--) {
				if(i == h && j == 0)
					branch = 0;

				for(int m = 0; m < k; m++) {
	    			Point p = {i, j, k};
	    			if(em(p).time == t) {
						if(em(p).thread.blockIdx != em(sp(p, branch, h)).thread.blockIdx) {
							onlyWithinBlock = 0;
							sbf = p.sbf;
							row = p.row;
							break;
						}
					}
	    		}
	    		if (onlyWithinBlock == 0)
				{
					SBF[a] = sbf;
					ROW[a] = row;
					a++;
				}
				onlyWithinBlock = 1;
	    		t++;
			}
			Point p = {i, 0, 0};
			range = sp(p, 1, h).row;
		}
		SBF[a] = h;
	    ROW[a] = 0;

	    /**for(int i = 0; i < tmax; i++)
	    	printf("[%d]: (%d, %d)\n", i, SBF[i], ROW[i]);

	    printf("\n======================\n");**/

	    /** PARALELL **/
	    int arraySize = sizeof(int) * threadsPerBlock;

        	cudaMemcpy(d_array, h_array, k * sizeof(int), cudaMemcpyHostToDevice);
	        gettimeofday(&t1, 0);
		    bitonicSortDDA<<<numBlocks, threadsPerBlock, arraySize>>>(d_array, 1, SBF[0], 0, ROW[0], h, k);
			if(SBF[0] < h || (SBF[0] == h && ROW[0] != 0)) {
				for(int j = 1; j <= a; j++) {
					// Kernel Instance
					Point p = {SBF[j - 1], ROW[j - 1], 0};
					bitonicSortDDA<<<numBlocks, threadsPerBlock, arraySize>>>(d_array, sp(p, 0, h).sbf, SBF[j], sp(p, 0, h).row, ROW[j], h, k);
				if(SBF[j] == 14 && ROW[j] == 0)
					printArrayGPU<<<1,1>>>(d_array, k);
				}
			cudaThreadSynchronize();
	        gettimeofday(&t2, 0);
    		elapsedTime += (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000000.0;
	    }
		//printArrayGPU<<<1,1>>>(d_array, k);
		checkArrayGPU<<<1,1>>>(d_array, k);
		printf("Bloques: %d Hilos por bloque: %d\n", numBlocks, threadsPerBlock);
		printf("N = %d - > Acabado Paralelo\n", k);
		//printArrayGPU<<<1,1>>>(d_array, k);

        /** >>>>>>> **/


        /** Sequentiall **/
        /**
        //cudaProfilerStart();
        for(int i = 0; i < REP_SEQ; i++) {
	        gettimeofday(&t1_seq, 0);
	        bitonicSortSec(h_array, k);
	        gettimeofday(&t2_seq, 0);
            elapsedTimeSec += (1000000.0*(t2_seq.tv_sec-t1_seq.tv_sec) + t2_seq.tv_usec-t1_seq.tv_usec)/1000000.0;

    	}
        //cudaProfilerStop();
        printf("N = %d - > Acabado Secuencial\n", k);

        /** >>>>>>> **/

		elapsedTime = elapsedTime / REP_PAR;
        //elapsedTimeSec = elapsedTimeSec / REP_SEQ;
        elapsedTimeSec = 2;
        float speedup = elapsedTimeSec / elapsedTime;

        fprintf(f1,"%d\t%d\t%d\t\t%.6fs\t%.6fs\t%.3f\n",k, numBlocks, threadsPerBlock, elapsedTime, elapsedTimeSec, speedup);
        inc *= 2;

        cudaFree(d_array);
    }

    cudaDeviceSynchronize();

    fclose(f1);

	return 0;
}