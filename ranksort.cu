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
const int N_MAX = 131072;
const int N = 8;
const int THREAD_MAX = 1024;
const int BLOCK_MAX = 65535;

void initArray(int *h_array, int k) {
	int mult = pow(10.0, LENGHT);
	for (int i = 0; i < k; i++){
	    h_array[i] = rand() % mult;
	}
}

void rankSortSeq(int * array, int * result, int k) {
	int rank;
	int miNumero;
	int comparador;
	for(int i = 0; i < k; i++) {
		rank = 0;
		miNumero = array[i];
		for(int j = 0; j < k; j++) {
			comparador = array[j];
			if(comparador < miNumero || (comparador == miNumero && (j < i)))
				rank++;
		}
		result[rank] = miNumero;
	}
}


__global__ void rankSort(int * array, int * result, int k) {// Ejemplo: 4 numeros; 2 bloques; 4 threadsPorBloque;
	int a = k / gridDim.x;
	int b = k / blockDim.x;
	__shared__ int tamBlocks;
	__shared__ int tamThreads;
	__shared__ int miNumero;
	__shared__ int rank;
	int localRank = 0;
	int comparador;
	int range2 = threadIdx.x * b;

	if(threadIdx.x == 0) {
		tamBlocks = k - (a * gridDim.x);
		tamThreads = k - (b * blockDim.x);
	}

	int range1 = blockIdx.x * a;
	for(int i = range1; i < range1 + a; i++) {	
		//printf("Relacion [talla/bloques]: %d; Ralacion [talla/threadsPerBlock]: %d\n", a, b );
		if(threadIdx.x == 0) {
			//sFlag = 0;
			miNumero = array[i];
			rank = 0;
			//printf("Soy el bloque %d y mi número es %d\n", blockIdx.x, miNumero);
		}
		__syncthreads();

		for(int j = range2; j < range2 + b; j++) {
			comparador = array[j];
			//printf("Soy el thread %d del bloque %d y mi comparador es %d\n contra %d", threadIdx.x, blockIdx.x, comparador, miNumero);
			if(comparador < miNumero || (comparador == miNumero && (j < i)))
				localRank += 1;
				//atomicAdd(&rank, 1);
		}

		if(threadIdx.x < tamThreads) {
			comparador = array[(blockDim.x * b) + threadIdx.x];
			if(comparador < miNumero || (comparador == miNumero && (((blockDim.x * b) + threadIdx.x) < i)))
				localRank += 1;
				//atomicAdd(&rank, 1);
		}

		atomicAdd(&rank, localRank);

		__syncthreads();

		if(threadIdx.x == 0) {
			result[rank] = miNumero;
		}

		__syncthreads();
	}

	// Compute the rest

	if(blockIdx.x < tamBlocks) {
		if(threadIdx.x == 0) {
			miNumero = array[gridDim.x * a  + blockIdx.x];
			rank = 0;
		}

		__syncthreads();

		for(int j = range2; j < range2 + b; j++) {
			comparador = array[j];
			if(comparador < miNumero || (comparador == miNumero && (j < (gridDim.x * a + blockIdx.x))))
				atomicAdd(&rank, 1);
		}

		if(threadIdx.x < tamThreads) {
			comparador = array[(blockDim.x * b) + threadIdx.x];
			if(comparador < miNumero || (comparador == miNumero && (((blockDim.x * b) + threadIdx.x) < gridDim.x * a  + blockIdx.x)))
				atomicAdd(&rank, 1);
		}

		__syncthreads();

		if(threadIdx.x == 0) {
			result[rank] = miNumero;
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
	FILE *f1 = fopen("rank_sort_2.dat", "w");
    fprintf(f1, "N\tBlocks\tThreads/Block\tT_Par\t\tT_Seq\t\tSpeed-Up\n");

	struct timeval t1, t2, t1_seq, t2_seq;
    double elapsedTime, elapsedTimeSec;

    int inc = 2;
    for(int k = N; k < N_MAX; k*=2) {
	    int numBlocks = k;
	    int threadsPerBlock = k;
	    
	    if(threadsPerBlock > THREAD_MAX)
	    	threadsPerBlock = THREAD_MAX;

		if(numBlocks > BLOCK_MAX)
			numBlocks = BLOCK_MAX;

	    int h_array[k];
	    int h_result[k];
	    int *d_array;
	    int *d_result;

	    initArray(h_array, k);

	    cudaMalloc(&d_array, k * sizeof(int));
	    cudaMalloc(&d_result, k * sizeof(int));
	    cudaMemcpy(d_array, h_array, k * sizeof(int), cudaMemcpyHostToDevice);


	    /** PARALELL **/
	    
	    gettimeofday(&t1, 0);
    	rankSort<<<numBlocks, threadsPerBlock>>>(d_array, d_result, k);
	    cudaThreadSynchronize();
        gettimeofday(&t2, 0);
        printf("N = %d - > Acabado Paralelo\n", k);

        /** >>>>>>> **/

        printArray<<<1,1>>>(d_result, k);

        /** SEQUENTIALL **/
        
        gettimeofday(&t1_seq, 0);
        rankSortSeq(h_array, h_result, k);
        gettimeofday(&t2_seq, 0);
        printf("N = %d - > Acabado Secuencial\n", k);
        printArray<<<1,1>>>(h_result, k);

        /** >>>>>>> **/

        elapsedTime = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000000.0;
        elapsedTimeSec = (1000000.0*(t2_seq.tv_sec-t1_seq.tv_sec) + t2_seq.tv_usec-t1_seq.tv_usec)/1000000.0;
        float speedup = elapsedTimeSec / elapsedTime;

        fprintf(f1,"%d\t%d\t%d\t\t%.6fs\t%.6fs\t%.3f\n",k, numBlocks, threadsPerBlock, elapsedTime, elapsedTimeSec, speedup);

        inc *= 2;

	    cudaFree(d_array);
	    cudaFree(d_result);
	    cudaDeviceSynchronize();
	}

	fclose(f1);

    return 0;
}