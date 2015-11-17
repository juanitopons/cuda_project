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
const int BLOCK_MAX = 65535;

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

    printf("Boolean = %d; (if 1 all OK)\n\n", boolean);
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

    printf("Boolean = %d; (if 1 all OK)\n\n", boolean);
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


///////////////////////////////
// CPU Sequentiall Algorithm //
///////////////////////////////

/**
 * [rankSortSeq description] Sequential ranksort algorithm function.
 *
 * @param array  [description] Array to be sorted.
 * @param result [description] Result array: it's the sorted final array.
 * @param k      [description] Size of the array.
 */
void rankSortSeq(int * array, int * result, int k) {
	int rank; // Rank (index) accumulation to sort an array position.
	int miNumero; // Number to RANK.
	int comparador; // Number to compare with the RANK (miNumero).
	for(int i = 0; i < k; i++) {
		rank = 0;
		miNumero = array[i];
		for(int j = 0; j < k; j++) {
			comparador = array[j];
			if(comparador < miNumero || (comparador == miNumero && (j < i)))
				rank++;
		}
		result[rank] = miNumero; // Placing the number in its sorted position
	}
}


////////////////////////////
// GPU Parallel algorithm //
////////////////////////////

/**
 * [rankSort description] GPU function. Parallel ranksort algorithm function.
 *
 * @param array  [description] Array to be sorted.
 * @param result [description] Result array: it's the sorted final array.
 * @param k      [description] Size of the array.
 */
__global__ void rankSort(int * array, int * result, int k) {
	int a = k / gridDim.x; // Relation: How many numbers have a block to RANK.
	int b = k / blockDim.x; // Relation: How many numbers have a thread to COMPARE with the RANK number.
	__shared__ int tamBlocks; // Extra (rest) number(s) that lower blocks will have to RANK after.
	__shared__ int tamThreads; // Extra (rest) number(s) that lower threads will have to COMPARE with the actual RANK.
	__shared__ int miNumero; // Number to RANK.
	__shared__ int rank; // Rank (index) accumulation to sort an array position.
	int localRank; // Local thread Rank (index) accumulation (will be sum to the global one at the end of the thread comparissions) 
	int comparador; // Number to compare with the RANK (miNumero).
	int range2 = threadIdx.x * b; // Second loop range distribution (comparissions)

	if(threadIdx.x == 0) {
		// Rest of the numbers (indexes) that don't fit with the block distribution to be RANK after with lower blocks IDS
		tamBlocks = k - (a * gridDim.x);
		tamThreads = k - (b * blockDim.x);
	}

	int range1 = blockIdx.x * a; // First loop range distribution (numbers to RANK)
	for(int i = range1; i < range1 + a; i++) {	
		if(threadIdx.x == 0) {
			miNumero = array[i]; // We get the RANK number so we will let threads to make their comparissions
			rank = 0; // Initial shared rank
		}
		__syncthreads();

		localRank = 0; // Initial thead local rank
		for(int j = range2; j < range2 + b; j++) {
			comparador = array[j];
			if(comparador < miNumero || (comparador == miNumero && (j < i)))
				localRank += 1; // Local rank accumulation
		}

		// Let the lower threads ID's compute the 'rest' of the comparissions //
		if(threadIdx.x < tamThreads) { 
			comparador = array[(blockDim.x * b) + threadIdx.x];
			if(comparador < miNumero || (comparador == miNumero && (((blockDim.x * b) + threadIdx.x) < i)))
				localRank += 1; // Local rank accumulation
		}

		atomicAdd(&rank, localRank); // Atomic shared rank accumulation

		__syncthreads();

		if(threadIdx.x == 0) {
			result[rank] = miNumero; // Placing the number in its sorted position
		}

		__syncthreads();
	}


	// Let the lower blocks ID's compute the 'rest' of the RANKS //
	if(blockIdx.x < tamBlocks) {
		if(threadIdx.x == 0) {
			miNumero = array[gridDim.x * a  + blockIdx.x];
			rank = 0;
		}

		__syncthreads();

		localRank = 0;
		for(int j = range2; j < range2 + b; j++) {
			comparador = array[j];
			if(comparador < miNumero || (comparador == miNumero && (j < (gridDim.x * a + blockIdx.x))))
				localRank += 1; // Local rank accumulation
		}

		// Let the lower threads ID's compute the 'rest' of the comparissions //
		if(threadIdx.x < tamThreads) {
			comparador = array[(blockDim.x * b) + threadIdx.x];
			if(comparador < miNumero || (comparador == miNumero && (((blockDim.x * b) + threadIdx.x) < gridDim.x * a  + blockIdx.x)))
				localRank += 1; // Local rank accumulation
		}

		atomicAdd(&rank, localRank); // Atomic shared rank accumulation

		__syncthreads();

		if(threadIdx.x == 0) {
			result[rank] = miNumero; // Placing the number in its sorted position
		}
	}
}


//////////
// MAIN //
//////////

int main( int argc, char* argv[] )
{
	// File for saving data times
	FILE *f1 = fopen("rank_sort_parallel.dat", "w");
    fprintf(f1, "N\tBlocks\tThreads/Block\tT_Par\t\tT_Seq\t\tSpeed-Up\n");

    // Time structures variables for sequentiall and parallel calculation
	struct timeval t1, t2, t1_seq, t2_seq;
    double elapsedTime = 0, elapsedTimeSec = 0;

    for(int k = N; k < N_MAX; k*=2) {

    	// Total number of blocks in the GPU
	    int numBlocks = k;

	    // Threads per each block in the GPU
	    int threadsPerBlock = k;
	    
	    if(threadsPerBlock > THREAD_MAX)
	    	threadsPerBlock = THREAD_MAX;

		if(numBlocks > BLOCK_MAX)
			numBlocks = BLOCK_MAX;

		// Elements data arrays (CPU and (h_) GPU (d_))
	    int h_array[k];
	    int h_result[k];
	    int *d_array;
	    int *d_result;

	    // Random array initialization
	    initArray(h_array, k);

	    // Array memory allocation
	    cudaMalloc(&d_array, k * sizeof(int));
	    cudaMalloc(&d_result, k * sizeof(int));


	    //////////////
        // PARALELL //
        //////////////
        
	    for(int m = 0; m < REP_PAR; m++) {
	    	cudaMemcpy(d_array, h_array, k * sizeof(int), cudaMemcpyHostToDevice);
		    gettimeofday(&t1, 0); // Started time
            cudaProfilerStart(); // Profiler start

	    	rankSort<<<numBlocks, threadsPerBlock>>>(d_array, d_result, k);

		    cudaThreadSynchronize(); // Synchronization
            cudaProfilerStop(); // Profiler stop
            gettimeofday(&t2, 0); // Stopped time
	        elapsedTime += (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000000.0;
	    }

        printf("N = %d - > Ended Paralell\n", k);
        checkArrayGPU<<<1,1>>>(d_result, k);


        /////////////////
        // Sequentiall //
        /////////////////

	    // Array copy auxiliar
        int h_array2[k];

        for(int m = 0; m < REP_SEQ; m++) {
            memcpy(h_array2, h_array, k * sizeof(int));
	        gettimeofday(&t1_seq, 0);
	        //cudaProfilerStart();
	        rankSortSeq(h_array2, h_result, k);
	        //cudaProfilerStop();
	        gettimeofday(&t2_seq, 0);
            elapsedTimeSec += (1000000.0*(t2_seq.tv_sec-t1_seq.tv_sec) + t2_seq.tv_usec-t1_seq.tv_usec)/1000000.0;
	    }

        printf("N = %d - > Ended Sequentiall\n", k);
        checkArrayCPU(h_result, k);

        
        ///////////////////////////////////
        // Times and SpeedUp calculation //
        ///////////////////////////////////

        elapsedTime = elapsedTime / REP_PAR;
        elapsedTimeSec = elapsedTimeSec / REP_SEQ;
        float speedup = elapsedTimeSec / elapsedTime;

 		fprintf(f1,"%d\t%d\t%d\t\t%.6fs\t%.6fs\t%.3f\n",k, numBlocks, threadsPerBlock, elapsedTime, elapsedTimeSec, speedup);

 		// Free GPU memory
	    cudaFree(d_array);
	    cudaFree(d_result);
	}

	cudaDeviceSynchronize();

	fclose(f1); // Close file

    return 0;
}