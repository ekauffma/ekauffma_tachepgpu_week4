#include <stdio.h>
#include <time.h>

const int DSIZE = 256;
const int a = 2;
const int b = 4;

// error checking macro
#define cudaCheckErrors()                                       \
	do {                                                        \
		cudaError_t __err = cudaGetLastError();                 \
		if (__err != cudaSuccess) {                             \
			fprintf(stderr, "Error:  %s at %s:%d \n",           \
			cudaGetErrorString(__err),__FILE__, __LINE__);      \
			fprintf(stderr, "*** FAILED - ABORTING***\n");      \
			exit(1);                                            \
		}                                                       \
	} while (0)


// CUDA kernel that runs on the GPU
__global__ void dot_product(const int *A, const int *B, int *C, int N) {

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        atomicAdd(C, A[idx] * B[idx]);
    }

}


int main() {

	// Create the device and host pointers
	int *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;

	// Fill in the host pointers 

	h_A = new int[DSIZE];
	h_B = new int[DSIZE];
	h_C = new int;
	for (int i = 0; i < DSIZE; i++){
		h_A[i] = a;
    	h_B[i] = b;
	}
	*h_C = 0;

	// Allocate device memory 
    cudaMalloc(&d_A, DSIZE * sizeof(int));
    cudaMalloc(&d_B, DSIZE * sizeof(int));
    cudaMalloc(&d_C, sizeof(int));
	
	// Check memory allocation for errors
    cudaCheckErrors();

	// Copy the matrices on GPU
    cudaMemcpy(d_A, h_A, DSIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, DSIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, sizeof(int), cudaMemcpyHostToDevice);
	
	// Check memory copy for errors
    cudaCheckErrors();

	// Define block/grid dimensions and launch kernel
    int blockSize = 32;
    int gridSize = DSIZE / blockSize;
    dot_product<<<gridSize, blockSize>>>(d_A, d_B, d_C, DSIZE);
	
	// Copy results back to host
    cudaMemcpy(h_A, d_A, DSIZE * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_B, d_B, DSIZE * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C, d_C, sizeof(int), cudaMemcpyDeviceToHost);

    // Check copy for errors
    cudaCheckErrors();

	// Verify result
    printf("h_A = [");
    for (int i=0; i < DSIZE; i++) {
        printf("%d, ",h_A[i]);
    }
    printf("]\n\n");
    printf("h_B = [");
    for (int i=0; i < DSIZE; i++) {
        printf("%d, ",h_B[i]);
    }
    printf("]\n\n");
    printf("h_C = %d \n\n", *h_C);

	// Free allocated memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
	
	return 0;

}
