__global__ void softmaxKernel(__nv_bfloat16 *input, int num_tokens) {
    __shared__ float row[1024]; 
    __shared__ float max_val;
    int workIndex = blockIdx.x * num_tokens + threadIdx.x;
    __nv_bfloat16 token = input[workIndex];
    row[threadIdx.x] = (float)token;
    __syncthreads();

    for (int i = 1; i < num_tokens; i = i * 2) {
        if (threadIdx.x % (i * 2) == 0 && threadIdx.x + i < num_tokens) {
            row[threadIdx.x] = fmaxf(row[threadIdx.x], row[threadIdx.x + i]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        max_val = row[0]; 
    }
    __syncthreads();

    row[threadIdx.x] = expf((float)token - max_val);
    __syncthreads();

    for (int i = 1; i < num_tokens; i = i * 2) {
        if (threadIdx.x % (i * 2) == 0 && threadIdx.x + i < num_tokens) {
            row[threadIdx.x] = row[threadIdx.x] + row[threadIdx.x + i];
        }
        __syncthreads();
    }

    input[workIndex] = (__nv_bfloat16)(expf((float)token - max_val) / row[0]);
}


void softmax(__nv_bfloat16 *input, int num_tokens)
{
    if (num_tokens > 1024)
    {
        std::cout << "Can't launch more than 1024 threads on RTX 5090, Softmax kernel not launched";
        return;
    }

    softmaxKernel<<<num_tokens * NUM_Q_HEADS, num_tokens>>>(input, num_tokens);
#ifdef DEBUG
    cudaError error = cudaGetLastError();
    if (error != cudaError::cudaSuccess)
    {
        std::cout << "CUDA last error: " << cudaGetLastError() << std::endl;
    }
#endif
}