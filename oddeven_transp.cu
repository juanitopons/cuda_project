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
const int N_MAX = 1048576;
const int N = 4;
const int THREAD_MAX = 1024;

void initArray(int *h_array, int k) {
	int mult = pow(10.0, LENGHT);
	for (int i = 0; i < k; i++){
	    h_array[i] = rand() % mult;
	}
}

void oddEven_transpSeq(int * h_array, int k) {
	for(int i = 1; i < k; i++) {
		if(i % 2 != 0) {
			//ODD
			for(int j = 0; j < (k / 2); j++) {
				int index = j * 2;
				if(h_array[index] > h_array[index + 1]) {
					int aux = h_array[index];
					h_array[index] = h_array[index + 1];
					h_array[index + 1] = aux;
				}
			}
		} else {
			//EVEN
			for(int j = 0; j < ((k - 1) / 2); j++) {
				int index = j * 2 + 1;
				if(h_array[index] > h_array[index + 1]) {
					int aux = h_array[index];
					h_array[index] = h_array[index + 1];
					h_array[index + 1] = aux;
				}
			}
		}
	}
}

__global__ void oddEven_transp_odd(int * d_array, int k) {
	int me = (blockIdx.x * blockDim.x + threadIdx.x);

	if(me < (k / 2)) {
		int indexA = me * (k / (blockDim.x * gridDim.x));
		int indexB = indexA + 1;
		int a = d_array[indexA];
		int b = d_array[indexB];

		//printf("Soy Thread %d del Bloque %d, mi índice es [%d] y [%d]: %d y %d\n", threadIdx.x, blockIdx.x, indexA, indexB, a, b);

		if(a > b) {
			d_array[indexA] = b;
			d_array[indexB] = a;
		}
	}
}

__global__ void oddEven_transp_even(int * d_array, int k) {
	int me = (blockIdx.x * blockDim.x + threadIdx.x);

	if(me < ((k - 1) / 2)) {
		int indexA = me * (k / (blockDim.x * gridDim.x)) + 1;
		int indexB = indexA + 1;
		int a = d_array[indexA];
		int b = d_array[indexB];

		if(a > b) {
			d_array[indexA] = b;
			d_array[indexB] = a;
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
	FILE *f1 = fopen("oddeven_transp_sort_1048576.dat", "w");
    fprintf(f1, "N\tBlocks\tThreads/Block\tT_Par\t\tT_Seq\t\tSpeed-Up\n");

	struct timeval t1, t2, t1_seq, t2_seq;
    double elapsedTime, elapsedTimeSec;

    int inc = 2;
    for(int k = N; k < N_MAX; k*=2) {
		int numBlocks = 1;
	    int threadsPerBlock = floor((double)k / 2);

	    int divisor = 2;
	    while(threadsPerBlock > THREAD_MAX) {
	    	int resto = threadsPerBlock % divisor;
	    	if(resto == 0) {
	    		threadsPerBlock = threadsPerBlock / divisor;
	    		numBlocks *= divisor;
	    		divisor = 2;
	    	} else {
	    		divisor++;
	    	}
	    }

	    int h_array[k];
	    int *d_array;

	    initArray(h_array, k);

	    cudaMalloc(&d_array, k * sizeof(int));
	    cudaMemcpy(d_array, h_array, k * sizeof(int), cudaMemcpyHostToDevice);


	 	/** PARALELL **/

	    gettimeofday(&t1, 0);

	    oddEven_transp_odd<<<numBlocks, threadsPerBlock>>>(d_array, k);
		for(int i = 1; i < k; i++) {
			if(i % 2 != 0)
				oddEven_transp_even<<<numBlocks, threadsPerBlock>>>(d_array, k);
			else
				oddEven_transp_odd<<<numBlocks, threadsPerBlock>>>(d_array, k);
		}

		/** >>>>>>> **/

		cudaThreadSynchronize();
	    gettimeofday(&t2, 0);
	    printf("N = %d - > Acabado Paralelo\n", k);
	    printArray<<<1,1>>>(d_array, k);


		/** SEQUENTIALL **/

	    gettimeofday(&t1_seq, 0);

	    oddEven_transpSeq(h_array, k);

	    gettimeofday(&t2_seq, 0);
	    printf("N = %d - > Acabado Secuencial\n", k);
	    printArray<<<1,1>>>(h_array, k);

		/** >>>>>>> **/

		elapsedTime = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000000.0;
        elapsedTimeSec = (1000000.0*(t2_seq.tv_sec-t1_seq.tv_sec) + t2_seq.tv_usec-t1_seq.tv_usec)/1000000.0;
        float speedup = elapsedTimeSec / elapsedTime;

        fprintf(f1,"%d\t%d\t%d\t\t%.6fs\t%.6fs\t%.3f\n",k, numBlocks, threadsPerBlock, elapsedTime, elapsedTimeSec, speedup);

        inc *= 2;

	    cudaFree(d_array);
	    cudaDeviceSynchronize();
	}

	fclose(f1);

    return 0;
}