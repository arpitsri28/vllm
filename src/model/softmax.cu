__global__ void softmaxKernel(__nv_bfloat16* input, int row_length) {
    __shared__ float row[1024];
    __shared__ float max_val;
    int workIndex = blockIdx.x * row_length + threadIdx.x;
    float token = (float)input[workIndex];
    row[threadIdx.x] = token;
    __syncthreads();
    for (int i = 1; i < row_length; i *= 2) {
        if ( threadIdx.x % (i * 2) == 0 && threadIdx.x + i < row_length) {
            row[threadIdx.x] =
                fmaxf(
                    row[threadIdx.x],
                    row[threadIdx.x + i]
                );
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        max_val = row[0];
    }

    __syncthreads();

    row[ threadIdx.x ] =
        expf(token - max_val);

    __syncthreads();

    for (int i = 1; i < row_length; i *= 2) {

        if ( threadIdx.x % (i * 2) == 0 && threadIdx.x + i < row_length) {
            row[threadIdx.x] = row[threadIdx.x] + row[threadIdx.x + i];
        }

        __syncthreads();
    }

    input[workIndex] = (__nv_bfloat16)(expf(token - max_val) / row[0]);
}

void softmaxRows(__nv_bfloat16* input, int num_rows, int row_length) {
    if (row_length > 1024) {
        std::cout
            << "Softmax row length exceeds 1024; kernel not launched"
            << std::endl;
        return;
    }

    softmaxKernel<<<num_rows, row_length>>>(
        input,
        row_length
    );

#ifdef DEBUG
    cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess) {
        std::cout
            << "CUDA last error: "
            << cudaGetErrorString(error)
            << std::endl;
    }
#endif
}