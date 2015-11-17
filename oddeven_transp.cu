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


///////////////////////////////
// CPU Sequentiall Algorithm //
///////////////////////////////

/**
 * [oddEven_transpSeq description] Sequential execution of OddEven transposition algorithm.
 *
 * @param h_array [description] CPU array to be sorted.
 * @param k       [description] Size of the array to be sorted.
 */
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


////////////////////////////
// GPU Parallel algorithm //
////////////////////////////

/**
 * [oddEven_transp_odd description] Odd execution of OddEven algorithm.
 *
 * @param d_array [description] Device (GPU) array to be sorted.
 * @param k       [description] Size of the array to be sorted.
 */
__global__ void oddEven_transp_odd(int * d_array, int k) {
	int me = (blockIdx.x * blockDim.x + threadIdx.x); // Thread unique ID

	if(me < (k / 2)) {
		int indexA = me * (k / (blockDim.x * gridDim.x));
		int indexB = indexA + 1;
		int a = d_array[indexA];
		int b = d_array[indexB];

		if(a > b) {
			d_array[indexA] = b;
			d_array[indexB] = a;
		}
	}
}

/**
 * [oddEven_transp_even description] Even execution of OddEven algorithm
 *
 * @param d_array [description] Device (GPU) array to be sorted
 * @param k       [description] Size of the array to be sorted
 */
__global__ void oddEven_transp_even(int * d_array, int k) {
	int me = (blockIdx.x * blockDim.x + threadIdx.x); // Thread unique ID

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


//////////
// MAIN //
//////////

int main( int argc, char* argv[] )
{
	// File for saving data times
	FILE *f1 = fopen("oddeven_transp_sort_parallel.dat", "w");
    fprintf(f1, "N\tBlocks\tThreads/Block\tT_Par\t\tT_Seq\t\tSpeed-Up\n");

    // Time structures variables for sequentiall and parallel calculation
	struct timeval t1, t2, t1_seq, t2_seq;
    double elapsedTime = 0, elapsedTimeSec = 0;

    for(int k = N; k < N_MAX; k*=2) {
		
		// Total number of blocks in the GPU
		int numBlocks = 1;

		// Threads per each block in the GPU
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

	    // Elements data arrays (CPU and (h_) GPU (d_))
	    int h_array[k];
	    int *d_array;

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

		    oddEven_transp_odd<<<numBlocks, threadsPerBlock>>>(d_array, k);
			for(int i = 1; i < k; i++) {
				if(i % 2 != 0)
					oddEven_transp_even<<<numBlocks, threadsPerBlock>>>(d_array, k);
				else
					oddEven_transp_odd<<<numBlocks, threadsPerBlock>>>(d_array, k);
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
		    oddEven_transpSeq(h_array2, k);
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