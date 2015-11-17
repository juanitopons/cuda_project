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

const int REP_PAR = 32; // Iterations for a given parallel execution size
const int REP_SEQ = 2; // Iterations for a given sequentiall execution size
const int LENGHT = 2; // Lenght of the random sequence numbers
const int N_MAX = 4194304; // End size
const int N = 1024; // Initial size
const int THREAD_MAX = 1024; // Max threads per block
int threadsPerBlock; // Number of threads to execute per block
int numBlocks; // Number of block to execute in the GPU

typedef unsigned int Branch;

/////////////
// Structs //
/////////////

/**
 * [ThreadT description] 
 */
struct ThreadT
{
	unsigned int blockIdx; /** block ID of the Thread in the GPU **/
	unsigned int threadIdx; /** ID of the Thread in the GPU **/
};

/**
 * [CudaT description] CUDA time space thread construction
 */
struct CudaT
{
	ThreadT thread; /** Specific Thread **/
	unsigned int time; /** Thread space time value **/
};

/**
 * [Point description] DDA point related to a spacific Thread.
 * The point will be embeded in our DDA-based construction.
 */
struct Point
{
	unsigned int sbf; /** sub-butterfly number where the Point is. **/
	unsigned int row; /** local row where the Point is. **/
	unsigned int col; /** global column where the Point is. **/
};

/////////////////////
// Usage Functions //
/////////////////////

/**
 * [flip description] flips the k^th bit (where bit 0 is the rightmost, least significant bit of n)
 * in the binary representation of the integer n.
 *
 * @param  k [description] bit position
 * @param  n [description] number to flip
 *
 * @return   [description] fliped number
 */
int flip(int k, int n) {
    return n ^ (1 << k);
}

/**
 * [d_flip description] flip function for GPU execution. Same as flip.
 *
 * @param  k [description] bit position
 * @param  n [description] number to flip
 *
 * @return   [description] flipped number
 */
__device__ int d_flip(int k, int n) {
    return n ^ (1 << k);
}

/**
 * [grow description] Grows (gets) the time of an specific Thread.
 *
 * @param  b [description] Actual sub-butterfly number of the Thread.
 * @param  r [description] Actual row number of the Thread.
 *
 * @return   [description] Next row to move.
 */
int grow(int b, int r) {
	return b * (b - 1) / 2 + b - r;
}

/**
 * [bit description] GPU function. Extracts the position^th bit from the binary representation
 * of the number x.
 *
 * @param  position [description] Position of the bit to extract.
 * @param  x        [description] Number to be extracted.
 *
 * @return          [description] Extracted bit.
 */
__device__ int bit(int position, int x) {
	return (x >> position) & 1;
}

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

    printf("Check Global Array (END): ");
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


/////////
// DDA //
/////////

/**
 * [sb_row description] Get the sub-butterfly and row from a space time Thread value.
 *
 * @param  time [description] Thread space time value.
 *
 * @return      [description] Point of that specific time value.
 */
Point sb_row(int time){
  Point p;
  long int temp = time;
  int i = 0;

   do {
         i++;
         temp -= i;
      }
   while (temp > 0);
   p.sbf = i;
   p.row = (i * (i + 1) >> 1) - time;

   return p;
}

/**
 * [d_sb_row description] GPU function. Same as sb_row.
 *
 * @param  time [description] Thread space time value.
 *
 * @return      [description] Point of that specific time value.
 */
__device__ Point d_sb_row(int time){
  Point p;
  long int temp = time;
  int i = 0;

   do {
         i++;
         temp -= i;
      }
   while (temp > 0);
   p.sbf = i;
   p.row = (i * (i + 1) >> 1) - time;

   return p;
}

/**
 * [sp description] Get the supply component (Point) for a given Point and Branch.
 *
 * @param  p [description] Actual DDA Point.
 * @param  b [description] Branch direction.
 * @param  h [description] Global bitonic height.
 *
 * @return   [description] Supply Point of a given Point.
 */
Point sp(Point p, Branch b, int h) {
	if((p.sbf != h) && (p.row == 0)) {
		if(b == 0) return {p.sbf + 1, p.sbf, p.col};
		else return {p.sbf + 1, p.sbf, flip(p.sbf, p.col)};

	} else if(((p.sbf == 1) && (p.row == 1)) ||
		(p.sbf > 1) && ((1 <= p.row) && (p.row <= (p.sbf - 1)))) {
		if(b == 0) return {p.sbf, p.row - 1, p.col};
		else return {p.sbf, p.row - 1, flip(p.row - 1, p.col)};
	}
}

/**
 * [d_sp description] GPU function. Get the supply component (Point) for a given Point and Branch.
 *
 * @param  p [description] Actual DDA point.
 * @param  b [description] Branch direction.
 * @param  h [description] Global bitonic height.
 *
 * @return {Point}[description] Supply component Point.
 */
__device__ Point d_sp(Point p, Branch b, int h) {
	if((p.sbf != h) && (p.row == 0)) {
		if(b == 0) return {p.sbf + 1, p.sbf, p.col};
		else return {p.sbf + 1, p.sbf, d_flip(p.sbf, p.col)};

	} else if(((p.sbf == 1) && (p.row == 1)) ||
		(p.sbf > 1) && ((1 <= p.row) && (p.row <= (p.sbf - 1)))) {
		if(b == 0) return {p.sbf, p.row - 1, p.col};
		else return {p.sbf, p.row - 1, d_flip(p.row - 1, p.col)};
	}
}

/**
 * [rg description] Get the request component (Point) for a given Point and Branch.
 *
 * @param  p [description] Actual DDA point.
 * @param  b [description] Branch direction.
 *
 * @return   [description] Request component Point.
 */
bool rg(Point p, Branch b) {
	return (p.row != 1 || p.sbf != 1);
}

/**
 * [d_rp description] GPU function. Get the request component (Point) for a given Point and Branch.
 *
 * @param  p [description] Actual DDA point.
 * @param  b [description] Branch direction.
 *
 * @return   [description] Request component Point.
 */
__device__ Point d_rp(Point p, Branch b) {
	if(((p.sbf > 1) && ((0 <= p.row) && (p.row <= (p.sbf - 2)))) ||
		((p.sbf == 1) && (p.row == 0))) {
		if(b == 0) return {p.sbf, p.row + 1, p.col};
		else return {p.sbf, p.row + 1, d_flip(p.row, p.col)};

	} else if((p.sbf > 1) && (p.row == p.sbf - 1)) {
		if(b == 0) return {p.sbf - 1, 0, p.col};
		else return {p.sbf - 1, 0, d_flip(p.row, p.col)};
	}
}

/**
 * [cond description] GPU function. Minimax condition. Gets the condition value for knowing if we
 * have to make a max or a min.
 *
 * @param  p [description] Actual Point.
 *
 * @return   [description] True (MIN) or False (MAX).
 */
__device__ bool cond(Point p) {
	return (bit(p.sbf, p.col) == bit(p.row, p.col));
}


///////////////////////////////////////
// CUDA Space Time Threads Embedding //
///////////////////////////////////////

/**
 * [em description] Embed (Gets) the next Point into our DDA based construction.
 *
 * @param  p [description] Actual Point.
 *
 * @return   [description] Cuda Space Time Thread.
 */
CudaT em(Point p) {
	ThreadT t = {(p.col / threadsPerBlock), (p.col % threadsPerBlock)};
	return {t, grow(p.sbf, p.row)};
}


////////////////////////////
// GPU Parallel Algorithm //
////////////////////////////

/**
 * [bitonicSortDDA description] GPU funciton. Executes the Bitonic Sort DDA-based algorithm in the
 * GPU.
 *
 * @param d_array     [description] Device (GPU) array of specific space time block to be processed.
 * @param out_d_array [description] Processed array of specific space time block.
 * @param start       [description] CUDA space time to start the execution.
 * @param end         [description] CUDA space time to end the execution. (included)
 * @param h           [description] Global bitonic height.
 * @param k           [description] Array size.
 */
__global__ void bitonicSortDDA(int * d_array, int * out_d_array, int start, int end, int h, int k) {
	extern __shared__ int local_array[];
	int x = blockIdx.x * blockDim.x + threadIdx.x;

	if(x < k) {
		// We first do the first iteration from the global memory and after we process the rest
		int time_l = start;
		Point p = d_sb_row(start); // Getting the Point of actual time
		p.col = x; // Setting the correct col
		int num1 = d_array[x];
		int num2 = d_array[d_rp(p, 1).col];

		if(bit(p.sbf, p.col) == bit(p.row, p.col)) {
			//MIN
			local_array[threadIdx.x] = min(num1, num2);
		} else {
			//MAX
			local_array[threadIdx.x] = max(num1, num2);
		}

		__syncthreads();

		time_l++;

		while(time_l <= end) {
			p = d_sp(p, 0, h);
			num1 = local_array[threadIdx.x];
			num2 = local_array[d_rp(p, 1).col % blockDim.x];

			int tmp;

			if(bit(p.sbf, p.col) == bit(p.row, p.col)) {
				//MIN
				tmp = min(num1, num2);
			} else {
				//MAX
				tmp = max(num1, num2);
			}

			__syncthreads();

			local_array[threadIdx.x] = tmp;

			__syncthreads();

			time_l++;

		}

	    out_d_array[x] = local_array[threadIdx.x];

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

int main( int argc, char* argv[] )
{
	// File for saving data times
    FILE *f1 = fopen("bitonic_sort_dda.dat", "w");
    fprintf(f1, "N\tBlocks\tThreads/Block\tT_Par\t\tT_Seq\t\tSpeed-Up\n");

    // Time structures variables for sequentiall and parallel calculation
    struct timeval t1, t2, t1_seq, t2_seq;
    double elapsedTime = 0, elapsedTimeSec = 0;

    for(int k = N; k < N_MAX; k*=2) {

    	// Threads per each block in the GPU
		threadsPerBlock = THREAD_MAX;
		if(THREAD_MAX > k)
			threadsPerBlock = k;

		// Total number of blocks in the GPU
    	numBlocks = k / threadsPerBlock;

    	// Bitonic height
		int h = log(k) / log(2);

		// Elements data arrays (CPU and (h_) GPU (d_))
		int h_array[k];
	    int * d_array;
	    int * out_d_array;

	    // Max times (iterations)
	    int tmax = (h * (h + 1)) / 2;

	    // Times array
	    int times[tmax];

	    // Index of 'times' array
		int a = 0;
		//Current thread time space time
		int t = 0;

		// Random array initialization
		initArray(h_array, k);

		// Array memory allocation
		cudaMalloc(&d_array, k * sizeof(int));
		cudaMalloc(&out_d_array, k * sizeof(int));


		/////////////////////////////////////////
		// Space Time Kernel Calls calculation //
		/////////////////////////////////////////

		int onlyWithinBlock = 1;

		for(int i = 0; i < tmax; i++) {
			for(int m = 0; m < k; m++) {
				Point p = sb_row(i);
				p.col = m;
				if(em(p).time == t) {
					if(em(p).thread.blockIdx != em(sp(p, 1, h)).thread.blockIdx) {
						onlyWithinBlock = 0;
						break; //If we found it, we don't need to keep going
					}
				}
			}

			if (onlyWithinBlock == 0)
			{
				times[a] = t;
				a++;
			}
			onlyWithinBlock = 1;
			t++;
		}
		times[a] = tmax;

	    printf("\n======================\n");


        //////////////
        // PARALELL //
        //////////////

	    int inst = 1;
	    int arraySize = sizeof(int) * threadsPerBlock;

	    for(int i = 0; i < REP_PAR; i++) {
        	cudaMemcpy(d_array, h_array, k * sizeof(int), cudaMemcpyHostToDevice);
	        gettimeofday(&t1, 0); // Started time
			cudaProfilerStart(); // Profiler start

	        //////////////////
	        // Kernel Calls //
	        //////////////////

	        // First call
		    bitonicSortDDA<<<numBlocks, threadsPerBlock, arraySize>>>(d_array, out_d_array, 1, times[0], h, k);
    	    cudaMemcpy(d_array, out_d_array, k * sizeof(int), cudaMemcpyDeviceToDevice);

    	    // Rest of the calls
			if(times[0] < tmax) {
				for(int j = 1; j <= a; j++) {
					inst++;
					// Kernel Call
					bitonicSortDDA<<<numBlocks, threadsPerBlock, arraySize>>>(d_array, out_d_array, times[j-1] + 1, times[j], h, k);
					cudaMemcpy(d_array, out_d_array, k * sizeof(int), cudaMemcpyDeviceToDevice);
				}
	    	}

	    	cudaThreadSynchronize(); // Synchronization
	    	cudaProfilerStop(); // Profiler stop
        	gettimeofday(&t2, 0); // Stopped time
			elapsedTime += (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000000.0;
		}

		printf("Calls to Kernel: %d\n", inst);
		printf("Blocks: %d; Threads per Block: %d; Total Threads: %d; Size %d\n", numBlocks, threadsPerBlock, numBlocks*threadsPerBlock, numBlocks*threadsPerBlock);
		printf("N = %d - > Ended Paralell\n", k);
		checkArrayGPU<<<1,1>>>(d_array, k);

		printf("\n----------------------\n");


        /////////////////
        // Sequentiall //
        /////////////////
        
        // Array copy auxiliar
		int h_array2[k];

        for(int i = 0; i < REP_SEQ; i++) {
        	memcpy(h_array2, h_array, k * sizeof(int));
	        gettimeofday(&t1_seq, 0);
	        //cudaProfilerStart(); // Profiler start
	        bitonicSortSec(h_array2, k);
	        //cudaProfilerStop(); // Profiler stop
	        gettimeofday(&t2_seq, 0);
            elapsedTimeSec += (1000000.0*(t2_seq.tv_sec-t1_seq.tv_sec) + t2_seq.tv_usec-t1_seq.tv_usec)/1000000.0;

    	}
        printf("Size = %d - > Ended Sequentiall\n", k);
        checkArrayCPU(h_array2, k);

        
        ///////////////////////////////////
        // Times and SpeedUp calculation //
        ///////////////////////////////////

		elapsedTime = elapsedTime / REP_PAR;
        elapsedTimeSec = elapsedTimeSec / REP_SEQ;
        float speedup = elapsedTimeSec / elapsedTime;
        fprintf(f1,"%d\t%d\t%d\t\t%.6fs\t%.6fs\t%.3f\n",k, numBlocks, threadsPerBlock, elapsedTime, elapsedTimeSec, speedup);

        // Free GPU memory
        cudaFree(d_array);
        cudaFree(out_d_array);
    }

    cudaDeviceSynchronize();

    fclose(f1); // Close file

	return 0;
}