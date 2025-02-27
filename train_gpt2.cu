/*
GPT-2 Transformer Neural Net trained in raw CUDA
Non-trivial notes to be aware of:

We are being clever in the backward pass to conserve memory.
In particular, all parameters use a += in the backward pass, so we
can later do gradient accumulation. But all activations have = instead of +=
because these are faster (just read, no write). This is okay for all activations
except for those in the residual stream, where the gradients have to add. We make
sure that those parts work out ok and that we do a += as necessary. E.g.,
the layernorms are connected to the residuals so we += in layernorm backward.
*/

#include <filesystem>
#include <cub/thread/thread_store.cuh>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/scatter.h>
#include <time.h>
#include <assert.h>
#include <float.h>
#include <string.h>
#include <unistd.h>
#include <assert.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include <memory>
#include <chrono>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/fill.h>
#include <thrust/transform.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cub/device/device_for.cuh>
#include <cub/iterator/cache_modified_input_iterator.cuh>
#include <cub/iterator/cache_modified_output_iterator.cuh>
#include <cuda/std/mdspan>
#include <cuda/atomic>

#include <iostream>
#include <fstream>


template <class T>
using pinned_vector = thrust::host_vector<
    T, thrust::mr::stateless_resource_allocator<
           T, thrust::system::cuda::universal_host_pinned_memory_resource>>;

void dump_tensor(float* ddata, const std::string& filename, std::size_t size) {
    // Open a binary file for writing
    std::filesystem::create_directories("outs");
    thrust::host_vector<float> data(size);
    thrust::copy_n(thrust::device_pointer_cast(ddata), size, data.begin());

    std::ofstream outFile("outs/" + filename, std::ios::binary);
    if (!outFile) {
        std::cerr << "Error: Could not open the file for writing!" << std::endl;
        return;
    }

    // Write the vector data to the binary file
    outFile.write(reinterpret_cast<const char*>(data.data()), size * sizeof(float));

    // Close the file
    outFile.close();
    if (!outFile.good()) {
        std::cerr << "Error: Could not write to the file correctly!" << std::endl;
    }
}

void dump_tensor(int* ddata, const std::string& filename, std::size_t size) {
    // Open a binary file for writing
    thrust::host_vector<int> data(size);
    thrust::copy_n(thrust::device_pointer_cast(ddata), size, data.begin());

    std::ofstream outFile("outs/" + filename, std::ios::binary);
    if (!outFile) {
        std::cerr << "Error: Could not open the file for writing!" << std::endl;
        return;
    }

    // Write the vector data to the binary file
    outFile.write(reinterpret_cast<const char*>(data.data()), size * sizeof(int));

    // Close the file
    outFile.close();
    if (!outFile.good()) {
        std::cerr << "Error: Could not write to the file correctly!" << std::endl;
    }
}




// ----------------------------------------------------------------------------
// CUDA utils

// convenience macro for calculating grid/block dimensions for kernels
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

// CUDA error checking
void cudaCheck(cudaError_t error, const char *file, int line) {
  if (error != cudaSuccess) {
    printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
           cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }
};
#define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))

// cuBLAS error checking
void cublasCheck(cublasStatus_t status, const char *file, int line)
{
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("[cuBLAS ERROR]: %d %s %d\n", status, file, line);
        exit(EXIT_FAILURE);
    }
}
#define cublasCheck(status) { cublasCheck((status), __FILE__, __LINE__); }

// cuBLAS workspace. Hardcoding to 32MiB but only Hopper needs 32, for others 4 is OK
static size_t cublaslt_workspace_size = 32 * 1024 * 1024;
static void* cublaslt_workspace = NULL;
static cublasComputeType_t cublas_compute_type;
cublasHandle_t cublas_handle;
cublasLtHandle_t cublaslt_handle;

namespace cg = cooperative_groups;


__device__ float warp_reduce(cg::thread_block_tile<32>& warp, float val) {
    return cg::reduce(warp, val, cg::plus<float>{});
}

// ----------------------------------------------------------------------------
// fread convenience utils, with nice handling of error checking using macros
// simple replace fopen, fread, fclose with fopenCheck, freadCheck, fcloseCheck

FILE *fopen_check(const char *path, const char *mode, const char *file, int line) {
    FILE *fp = fopen(path, mode);
    if (fp == NULL) {
        fprintf(stderr, "Error: Failed to open file '%s' at %s:%d\n", path, file, line);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", line);
        fprintf(stderr, "  Path: %s\n", path);
        fprintf(stderr, "  Mode: %s\n", mode);
        exit(EXIT_FAILURE);
    }
    return fp;
}

#define fopenCheck(path, mode) fopen_check(path, mode, __FILE__, __LINE__)

void fread_check(void *ptr, size_t size, size_t nmemb, FILE *stream, const char *file, int line) {
    size_t result = fread(ptr, size, nmemb, stream);
    if (result != nmemb) {
        if (feof(stream)) {
            fprintf(stderr, "Error: Unexpected end of file at %s:%d\n", file, line);
        } else if (ferror(stream)) {
            fprintf(stderr, "Error: File read error at %s:%d\n", file, line);
        } else {
            fprintf(stderr, "Error: Partial read at %s:%d. Expected %zu elements, read %zu\n",
                    file, line, nmemb, result);
        }
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", line);
        fprintf(stderr, "  Expected elements: %zu\n", nmemb);
        fprintf(stderr, "  Read elements: %zu\n", result);
        exit(EXIT_FAILURE);
    }
}

#define freadCheck(ptr, size, nmemb, stream) fread_check(ptr, size, nmemb, stream, __FILE__, __LINE__)

void fclose_check(FILE *fp, const char *file, int line) {
    if (fclose(fp) != 0) {
        fprintf(stderr, "Error: Failed to close file at %s:%d\n", file, line);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", line);
        exit(EXIT_FAILURE);
    }
}

#define fcloseCheck(fp) fclose_check(fp, __FILE__, __LINE__)

// ----------------------------------------------------------------------------
// malloc error-handling wrapper util

void *malloc_check(size_t size, const char *file, int line) {
    void *ptr = malloc(size);
    if (ptr == NULL) {
        fprintf(stderr, "Error: Memory allocation failed at %s:%d\n", file, line);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", line);
        fprintf(stderr, "  Size: %zu bytes\n", size);
        exit(EXIT_FAILURE);
    }
    return ptr;
}

#define mallocCheck(size) malloc_check(size, __FILE__, __LINE__)

// ----------------------------------------------------------------------------
// all the kernels

// really bad naive kernel with atomicAdd
__global__ void encoder_backward_kernel(float* dwte, float* dwpe,
                                        const float* dout, const int* inp,
                                        int B, int T, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = B * T * C;

    if (idx < N) {
        int bt = idx / C;
        int b = bt / T;
        int t = bt % T;
        int c = idx % C;

        int ix = inp[b * T + t];

        const float* dout_btc = dout + b * T * C + t * C + c;
        float* dwte_ix = dwte + ix * C + c;
        float* dwpe_tc = dwpe + t * C + c;

        cuda::atomic_ref<float, cuda::thread_scope_device> dwte_ix_ref(*dwte_ix);
        cuda::atomic_ref<float, cuda::thread_scope_device> dwte_tc_ref(*dwpe_tc);

        dwte_ix_ref.fetch_add(*dout_btc, cuda::memory_order_relaxed);
        dwte_tc_ref.fetch_add(*dout_btc, cuda::memory_order_relaxed);
    }
}

__global__ void layernorm_forward_kernel3(float* __restrict__ out, float* __restrict__ mean, float* __restrict__ rstd,
                                    const float*  __restrict__ inp, const float*  __restrict__ weight,
                                    const float* __restrict__ bias, int N, int C) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    int idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();
    if(idx >= N) {
        return;
    }

    // the row of input that this group of threads is responsible for
    const float* x = inp + idx * C;

    // mean
    float sum = 0.0f;
    for (int i = warp.thread_rank(); i < C; i += warp.size()) {
        sum += x[i];
    }
    sum = warp_reduce(warp, sum);
    //sum = cg::reduce(warp, sum, cg::plus<float>{});
    float m = sum / C;
    if(warp.thread_rank() == 0 && mean != nullptr) {
        __stcs(mean + idx, m);
    }

    // rstd
    sum = 0.0f;
    for (int i = warp.thread_rank(); i < C; i += warp.size()) {
        float diff = x[i] - m;
        sum += diff * diff;
    }
    //sum = cg::reduce(warp, sum, cg::plus<float>{});
    sum = warp_reduce(warp, sum);
    float s = rsqrtf(sum / C + 1e-5f);
    if(warp.thread_rank() == 0 && rstd != nullptr) {
        __stcs(rstd + idx, s);
    }

    // final normalization and scaling by weight/bias
    float* o = out + idx * C;
    for (int c = warp.thread_rank(); c < C; c += warp.size()) {
        // load and store using the .cs "streaming" hint to the compiler,
        // indicating that this data will not be reused soon, and can be streamed through the caches
        // this allows the threads to get more cache-hits for the (shared) weight and bias parameters
        float n = s * (__ldcs(x+c) - m);
        __stcs(o+c, n * weight[c] + bias[c]);
    }
}

__global__ void permute_kernel(float* q, float* k, float* v,
                               const float* inp,
                               int B, int N, int NH, int d) {
    // okay so now, this kernel wants Q,K,V to all be of shape (B, NH, N, d)
    // but instead, we have a single tensor QKV (inp) of shape (B, N, 3, NH, d)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Q[b][nh_][n][d_] = inp[b][n][0][nh_][d_]
    if (idx < B * NH * N * d) {
        int b = idx / (NH * N * d);
        int rest = idx % (NH * N * d);
        int nh_ = rest / (N * d);
        rest = rest % (N * d);
        int n = rest / d;
        int d_ = rest % d;
        int inp_idx = (b * N * 3 * NH * d) + (n * 3 * NH * d) + (0 * NH * d) + (nh_ * d) + d_;
        q[idx] = __ldcs(&inp[inp_idx]);
        k[idx] = __ldcs(&inp[inp_idx + NH * d]);
        v[idx] = __ldcs(&inp[inp_idx + 2 * (NH * d)]);
    }
}

__global__ void permute_kernel_backward(float* dinp,
                                        const float* dq, const float* dk, const float* dv,
                                        int B, int N, int NH, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B * NH * N * d) {
        int b = idx / (NH * N * d);
        int rest = idx % (NH * N * d);
        int nh_ = rest / (N * d);
        rest = rest % (N * d);
        int n = rest / d;
        int d_ = rest % d;

        int inp_idx = (b * N * 3 * NH * d) + (n * 3 * NH * d) + (0 * NH * d) + (nh_ * d) + d_;
        dinp[inp_idx] = dq[idx];
        dinp[inp_idx + NH * d] = dk[idx];
        dinp[inp_idx + 2 * (NH * d)] = dv[idx];
    }
}

__global__ void unpermute_kernel(float* inp, float *out, int B, int N, int NH, int d) {
   // out has shape (B, nh, N, d) but we need to unpermute it to (B, N, nh, d)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // out[b][n][nh_][d_] <- inp[b][nh_][n][d_]
    if (idx < B * NH * N * d) {
        int b = idx / (NH * N * d);
        int rest = idx % (NH * N * d);
        int nh_ = rest / (N * d);
        rest = rest % (N * d);
        int n = rest / d;
        int d_ = rest % d;
        int other_idx = (b * NH * N * d) + (n * NH * d) + (nh_ * d) + d_;
        out[other_idx] = __ldcs(&inp[idx]);
    }
}

__global__ void unpermute_kernel_backward(float* dinp, const float *dout, int B, int N, int NH, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B * NH * N * d) {
        int b = idx / (NH * N * d);
        int rest = idx % (NH * N * d);
        int nh_ = rest / (N * d);
        rest = rest % (N * d);
        int n = rest / d;
        int d_ = rest % d;
        int other_idx = (b * NH * N * d) + (n * NH * d) + (nh_ * d) + d_;
        dinp[idx] = dout[other_idx];
    }
}

__device__ float& vec_at(float4& vec, int index) {
    return reinterpret_cast<float*>(&vec)[index];
}

__device__ float vec_at(const float4& vec, int index) {
    return reinterpret_cast<const float*>(&vec)[index];
}

__global__ void softmax_forward_kernel5(float* out, float inv_temperature, const float* inp, int N, int T) {
    // inp, out shape: (N, T, T), where N = B * NH
    // fuses the multiplication by scale inside attention
    // directly autoregressive, so we only compute the lower triangular part
    // uses the online softmax algorithm
    assert(T % 4  == 0);
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    // micro-optimization: we iterate backwards so that
    // after the softmax backward operation completes, the cache retains the
    // part of the matrix close to the upper left corner, which benefits the
    // matmul operation that immediately follows.
    // int idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank(); // forward order
    int idx = (gridDim.x - blockIdx.x - 1) * warp.meta_group_size() + warp.meta_group_rank(); // backward order
    if(idx >= N * T) {
        return;
    }
    int own_pos = idx % T;
    int pos_by_4 = own_pos / 4;

    // one row of inp, i.e. inp[idx, :] of shape (T,)
    const float* x = inp + idx * T;

    // not INF, so we don't get NaNs accidentally when subtracting two values.
    float maxval = -FLT_MAX;
    float sumval = 0.0f;

    const float4* x_vec = reinterpret_cast<const float4*>(x);
    for (int i = warp.thread_rank(); i < pos_by_4; i += warp.size()) {
        float4 v = x_vec[i];
        float old_maxval =  maxval;
        for(int k = 0; k < 4; ++k) {
            maxval = fmaxf(maxval, vec_at(v, k));
        }
        sumval *= expf(inv_temperature * (old_maxval - maxval));
        for(int k = 0; k < 4; ++k) {
            sumval += expf(inv_temperature * (vec_at(v, k) - maxval));
        }
    }

    if(4*pos_by_4 + warp.thread_rank() <= own_pos) {
        float old_maxval = maxval;
        maxval = fmaxf(maxval, x[4*pos_by_4 + warp.thread_rank()]);
        sumval *= expf(inv_temperature * (old_maxval - maxval));
        sumval += expf(inv_temperature * (x[4*pos_by_4 + warp.thread_rank()] - maxval));
    }

    float global_maxval = cg::reduce(warp, maxval, cg::greater<float>{});
    sumval *= expf(inv_temperature * (maxval - global_maxval));

    float sum = warp_reduce(warp, sumval);
    float norm = 1.f / sum;
    // divide the whole row by the sum
    for (int i = warp.thread_rank(); i <= own_pos; i += warp.size()) {
        // recalculation is faster than doing the round-trip through memory.
        float ev = expf(inv_temperature * (__ldcs(x + i) - global_maxval));
        __stcs(out + idx * T + i, ev * norm);
    }
}

#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)
__global__ void gelu_backward_kernel(float* dinp, const float* inp, const float* dout, const int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float x = inp[i];
        float cube = 0.044715f * x * x * x;
        float tanh_arg = GELU_SCALING_FACTOR * (x + cube);
        float tanh_out = tanhf(tanh_arg);
        float coshf_out = coshf(tanh_arg);
        float sech_out = 1.0f / (coshf_out * coshf_out);
        float local_grad = 0.5f * (1.0f + tanh_out) + x * 0.5f * sech_out * GELU_SCALING_FACTOR * (1.0f + 3.0f * 0.044715f * x * x);
        dinp[i] = local_grad * dout[i];
    }
}

__global__ void softmax_forward_kernel7(float* out, const float* inp, int N, int C) {
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    // out is (N, C) just like inp. Each row of inp will get softmaxed.
    // same as kernel4, but optimised for very large Cs with advanced unrolling

    // The trick is to read into a register array (all indices known at compile time)
    // and always read UNROLL_FACTOR values to maximise memory level parallelism
    // even if we would be out of bounds, we set the index to min(C-1, idx)
    // so we just do some unnecessary reads (obviously bad for small C)
    // the writes are in a separate loop with a conditional check for out of bounds
    // making it separate is necessary to convince the compiler to do the right thing
    const int UNROLL_FACTOR = 8;
    const int warpsPerBlock = blockDim.x / 32;

    extern __shared__ float shared[];
    int idx = cg::this_grid().block_rank();
    int tid = block.thread_rank();
    int warpId = warp.meta_group_rank(); // warp index within a block
    int laneId = warp.thread_rank(); // thread index within a warp

    // shared[] must be allocated to have 2 * warpsPerBlock elements
    // first half for max values, the second half for sum values
    float* maxvals = shared;
    float* sumvals = &shared[warpsPerBlock];

    if (tid >= C) {
        maxvals[warpId] = -INFINITY;
        sumvals[warpId] = 0.0f;
        return;
    }

    const float* x = inp + idx * C; // input
    float* y = out + idx * C; // output

    // first, thread coarsening by directly accessing global memory in series
    float maxval = -INFINITY;
    for (int i = tid; i < C; i += blockDim.x * UNROLL_FACTOR) {
        #pragma unroll
        for (int u = 0; u < UNROLL_FACTOR; u++) {
            maxval = fmaxf(maxval, x[min(C - 1, i + u * block.size())]);
        }
    }

    // now within-warp reductions for maxval
    maxval = cg::reduce(warp, maxval, cg::greater<float>{});
    // the 0th thread of each warp writes the maxval of that warp to shared memory
    if (laneId == 0) maxvals[warpId] = maxval;
    __syncthreads();
    // now the 0th thread reduces the maxvals in shared memory, i.e. across warps
    if (tid == 0) {
        float val = maxvals[tid];
        #pragma unroll
        for (int i = 1; i < warpsPerBlock; i++) {
            val = fmaxf(val, maxvals[i]);
        }
        // store the final max in the first position
        maxvals[0] = val;
    }
    __syncthreads();
    // broadcast the max to all threads
    float offset = maxvals[0];

    // compute expf and write the result to global memory
    // + thread coarsening for sum
    float sumval = 0.0f;
    for (int i = tid; i < C; i += block.size() * UNROLL_FACTOR) {
        float reg_array[UNROLL_FACTOR];
        #pragma unroll
        for (int u = 0; u < UNROLL_FACTOR; u++) {
            reg_array[u] = __ldcs(&x[min(C - 1, i + u * block.size())]);
        }
        #pragma unroll
        for (int u = 0; u < UNROLL_FACTOR; u++) {
            if (i + u * block.size() < C) {
                float output = expf(reg_array[u] - offset);
                y[min(C - 1, i + u * block.size())] = output; // compiler likes redundant min()?!
                sumval += output; // combined into the same loop unlike kernel3
            }
        }
    }

    // okay now we calculated exp(x - max(x))
    // step 2: sum all the values and divide by the sum

    // within-warp reduction for sumval
    //sumval = cg::reduce(warp, sumval, cg::plus<float>{});
    sumval = warp_reduce(warp, sumval);

    // write sumval to shared memory
    if (laneId == 0) sumvals[warpId] = sumval;
    __syncthreads();
    // inter-thread reduction of sum
    if (tid == 0) {
        float val = sumvals[tid];
        #pragma unroll
        for (int i = 1; i < warpsPerBlock; ++i) {
            val += sumvals[i];
        }
        sumvals[0] = val;
    }
    __syncthreads();
    // broadcast the sum to all threads
    float sum = sumvals[0];

    // divide the whole row by the sum
    for (int i = tid; i < C; i += block.size() * UNROLL_FACTOR) {
        float reg_array[UNROLL_FACTOR];
        #pragma unroll
        for (int u = 0; u < UNROLL_FACTOR; u++) {
            reg_array[u] = y[min(C - 1, i + u * block.size())];
        }
        #pragma unroll
        for (int u = 0; u < UNROLL_FACTOR; u++) {
            if (i + u * block.size() < C) {
                y[i + u * block.size()] = reg_array[u] / sum;
            }
        }
    }
}

// cooperative groups solution, one warp per output channel
__global__ void matmul_backward_bias_kernel2(float* dbias, const float* dout, int B, int T, int OC) {
    // dout is (B, T, OC), dbias is (OC)
    // e.g. if block_size = 128, then we have 4 warps per block, each in charge of one output channel
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    // meta_group_size is the number of warps in a block (e.g. 4), meta_group_rank is the warp index (0,1,2,3)
    int idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();
    if(idx >= OC) { return; }
    int BT = B * T; // number of elements to reduce in total, per channel
    // first, thread coarsening to sum reduce the problem size from B*T to 32
    float sum = 0.0f;
    for(int i = warp.thread_rank(); i < BT; i += warp.size()) {
        sum += dout[i * OC + idx];
    }
    // now do a warp-level reduce to get the sum across the 32 threads in this warp
    //sum = cg::reduce(warp, sum, cg::plus<float>{});
    sum = warp_reduce(warp, sum);
    // write the result to output (global memory)
    if(warp.thread_rank() == 0) {
        dbias[idx] += sum;
    }
}

__global__ void layernorm_backward_kernel(float* dinp, float* dweight, float* dbias,
                        const float* dout, const float* inp, const float* weight, const float* mean, const float* rstd,
                        int B, int T, int C) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    int idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();
    int N = B * T;
    if(idx >= N) {
        return;
    }

    int b = idx / T;
    int t = idx % T;

    const float* dout_bt = dout + b * T * C + t * C;
    const float* inp_bt = inp + b * T * C + t * C;
    float* dinp_bt = dinp + b * T * C + t * C;
    float mean_bt = mean[b * T + t];
    float rstd_bt = rstd[b * T + t];
    //if (blockIdx.x == 0 && warp.meta_group_rank() == 0) printf("%d Mean is %f\n",warp.thread_rank(), mean_bt);
    // first: two reduce operations
    float dnorm_mean = 0.0f;
    float dnorm_norm_mean = 0.0f;
    for (int i = warp.thread_rank(); i < C; i  += warp.size()) {
        float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
        float dnorm_i = weight[i] * dout_bt[i];
        dnorm_mean += dnorm_i;
        dnorm_norm_mean += dnorm_i * norm_bti;
    }

    dnorm_mean = warp_reduce(warp, dnorm_mean);
    dnorm_norm_mean = warp_reduce(warp, dnorm_norm_mean);
    dnorm_mean = dnorm_mean / C;
    dnorm_norm_mean = dnorm_norm_mean / C;

    // now iterate again and accumulate all the gradients
    for (int i = warp.thread_rank(); i < C; i += warp.size()) {
        float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
        float dnorm_i = weight[i] * dout_bt[i];

        cuda::atomic_ref<float, cuda::thread_scope_device> dbias_ref(dbias[i]);
        cuda::atomic_ref<float, cuda::thread_scope_device> dweight_ref(dweight[i]);

        // gradient contribution to bias
        dbias_ref.fetch_add(dout_bt[i], cuda::memory_order_relaxed);
        // gradient contribution to weight
        dweight_ref.fetch_add(norm_bti * dout_bt[i], cuda::memory_order_relaxed);

        //atomic_ordered_add(&dbias[i], dout_bt[i]);
        //atomic_ordered_add(&dweight[i], norm_bti * dout_bt[i]);

        // gradient contribution to input
        float dval = 0.0f;
        dval += dnorm_i; // term 1
        dval -= dnorm_mean; // term 2
        dval -= norm_bti * dnorm_norm_mean; // term 3
        dval *= rstd_bt; // final scale
        dinp_bt[i] += dval;
    }
}

__global__ void softmax_autoregressive_backward_kernel(float* dpreatt, const float* datt, const float* att,
                                                       int B, int T, int C, float scale) {
    constexpr const int BlockSize = 256;
    constexpr int T_per_block = 4;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    __shared__ float block_acc[32];

    int idx = blockIdx.y;
    // go through blocks in reverse order, so the slowest block starts first
    int t0 = T - 1 - T_per_block*blockIdx.x;

    //if (blockIdx.x == 3 && blockIdx.y == 4) {
    //    printf("%d %d %d %d %d %d %d\n",threadIdx.x, blockDim.x,blockDim.y, warp.meta_group_size(), warp.meta_group_rank(), warp.thread_rank(), block.thread_rank());
    //}
    att += idx * T * T;
    datt += idx * T * T;
    dpreatt += idx * T * T;

    if (warp.meta_group_rank() == 0) {
        block_acc[warp.thread_rank()] = 0;
    }

    for(int to = 0; to < T_per_block; ++to) {
        int t = t0 - to;
        if(t < 0) return;
        const float* att_bth = att + t * T;
        const float* datt_bth = datt + t * T;
        float* dpreatt_bth = dpreatt + t * T;

        float local_sum = 0;
        for (int t2 = block.thread_rank(); t2 <= t; t2 += BlockSize) {
            local_sum += att_bth[t2] * datt_bth[t2];
            //if (blockIdx.x == 0 && blockIdx.y ==0 && threadIdx.x == 0) printf("%d %f %f %f\n", idx * T * T + t*T + t2, local_sum, att_bth[t2], datt_bth[t2]); 
        }

        //if (blockIdx.x == 0 && blockIdx.y ==0 && warp.meta_group_rank() == 0) printf("%d local sum %f\n", threadIdx.x, local_sum);
        block_acc[warp.meta_group_rank()] = warp_reduce(warp, local_sum);
        //block_acc[warp.meta_group_rank()] = cg::reduce(warp, local_sum, cg::plus<float>{});
        block.sync();
        //if (blockIdx.x == 0 && blockIdx.y ==0 && threadIdx.x == 0) printf("local sum %f\n", block_acc[warp.thread_rank()]);
        //local_sum = cg::reduce(warp, block_acc[warp.thread_rank()], cg::plus<float>{});
        local_sum = warp_reduce(warp, block_acc[warp.thread_rank()]);
        for (int t3 = block.thread_rank(); t3 <= t; t3 += BlockSize) {
            // don't touch the cache. Some parts will still be here from the previous loop, and
            // we want to exploit those.
            float acc = __ldcs(att_bth + t3) * (__ldcs(datt_bth + t3) - local_sum);
            __stcs(dpreatt_bth + t3, scale * acc);
        }
    }
}

// Implements linear interpolation using only two floating-point operations (as opposed to three in a naive implementation).
// Reference: https://developer.nvidia.com/blog/lerp-faster-cuda
__device__ inline float lerp_(float start, float end, float weight) {
    return fma(weight, end, fma(-weight, start, start));
}

__global__ void adamw_kernel2(float* params_memory, float* grads_memory, float* m_memory, float* v_memory, long num_parameters,
                              float learning_rate, float beta1, float beta2, float beta1_correction, float beta2_correction, float eps, float weight_decay) {
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i >= num_parameters) return;  // guard
   float grad = grads_memory[i];
   float m = m_memory[i];
   float v = v_memory[i];
   // update the first moment (momentum)
   m = lerp_(grad, m, beta1);
   m_memory[i] = m;
   // update the second moment (RMSprop)
   v = lerp_(grad * grad, v, beta2);
   v_memory[i] = v;
   m /= beta1_correction;  // m_hat
   v /= beta2_correction;  // v_hat
   params_memory[i] -= learning_rate * (m / (sqrtf(v) + eps) + weight_decay * params_memory[i]);
}

struct SoftmaxParams {
    float Scale;
    float Offset;
};


__device__ SoftmaxParams prepare_softmax_blockwide_nofloat4(cg::thread_block_tile<32>& warp,
                                                   int idx, const float* inp, int V, int P) {
    // same but not float4
    // one row of inp, i.e. inp[idx, :] of shape (V,)

    const float* x = inp + idx * P;
    float thread_maxval = -INFINITY;
    float thread_sumval = 0.0f;
    // do the loop in reverse to maximise probability of L2 cache hits
    // so even small L2s get some hits on the 2nd read of the same thread
    for (int i = V + threadIdx.x - blockDim.x; i >= 0; i -= blockDim.x) {
        float v = x[i];
        float old_maxval = thread_maxval;
        thread_maxval = fmaxf(thread_maxval, v);
        thread_sumval *= expf((old_maxval - thread_maxval));
        thread_sumval += expf(v - thread_maxval);
    }

    // two reductions of up to 1024 threads:
    // 1) inside warp (shuffle), 2) cross-warp (shared memory), 3) inside warp (shuffle)
    // this results in much cleaner assembly than a multi-warp cg::reduce
    __shared__ float shared_maxval[32];
    __shared__ float shared_sumval[32];
    int num_warps = blockDim.x / 32;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    // reduce maxval within each warp
    float warp_maxval = cg::reduce(warp, thread_maxval, cg::greater<float>{});
    // thread 0 in each warp writes to shared memory
    if (lane_id == 0) { shared_maxval[warp_id] = warp_maxval; }
    __syncthreads();
    // each thread now loads the maxval across previous warps
    // if the thread is "out of range" of data, use -FLT_MAX as the maxval
    warp_maxval = (lane_id < num_warps) ? shared_maxval[lane_id] : -FLT_MAX;
    // now reduce the maxval among the warp threads
    float block_maxval = cg::reduce(warp, warp_maxval, cg::greater<float>{});
    // each thread uses maxval to scale sumval to avoid numerical instability / overflow
    thread_sumval *= expf(thread_maxval - block_maxval);
    // (warp-level) reduce sumval, thread 0 in each warp saves result in shared memory
    //float warp_sumval = cg::reduce(warp, thread_sumval, cg::plus<float>{});
    float warp_sumval = warp_reduce(warp, thread_sumval);
    if (lane_id == 0) { shared_sumval[warp_id] = warp_sumval; }
    __syncthreads();
    // same strategy, now reduce sumval across warps
    warp_sumval = (lane_id < num_warps) ? shared_sumval[lane_id] : 0.0f;
    //float block_sumval = cg::reduce(warp, warp_reduceval, cg::plus<float>{});
    float block_sumval =warp_reduce(warp, warp_sumval);
    // return the softmax parameters
    return SoftmaxParams{1.f / block_sumval, block_maxval};
}

// same as 2 but not using float4 (see dev/cuda/classifier_fused.cu)
// will _update_ logits to logit gradients
__global__ void fused_classifier_kernel3(float* logits, float* losses, float* probs,
                                         const float* dlosses, const int* targets,
                                         int B, int T, int V, int P) {
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    int idx = blockIdx.x;
    int ix = targets[idx];

    // softmax (reading B * T * V, same logits read again below, hopefully still in cache)
    SoftmaxParams sp = prepare_softmax_blockwide_nofloat4(warp, idx, logits, V, P);

    // calculate the probability needed for the loss and update (single-threaded)
    if(threadIdx.x == 0) {
        float prob = expf(logits[idx * P + ix] - sp.Offset) * sp.Scale;
        losses[idx] = -logf(prob);
    }

    // very sensible default for dlosses is 1/(B*T), which is the uniform loss
    float dloss = dlosses != NULL ? dlosses[idx] : 1.0f / (B*T);
    // calculate the gradients directly, saves bandwidth from probs during training
    // but also supports writing probs for inference-only and debugging
    const float* logits_vec = logits + idx * P;
    for (int i = threadIdx.x; i < V; i += blockDim.x) {
        // this is the 2nd read of logits after the one in prepare_softmax2
        // this data will never be needed again, so we reduce cache persistence
        float v = __ldcs(&logits_vec[i]);
        float prob = expf(v - sp.Offset) * sp.Scale;
        if (probs != NULL) {
            probs[idx * P + i] = prob;
        }
        float indicator = (i == ix) ? 1.0f : 0.0f;
        logits[idx * P + i] = (prob - indicator) * dloss;
    }
}

// ----------------------------------------------------------------------------
// kernel launchers

__device__ cuda::std::pair<int, int> i2n(int idx, int E1) {
    return {idx / E1, idx % E1};
}

__device__ cuda::std::tuple<int, int, int> i2n(int idx, int E1, int E2) {
    int bt = idx / E1;
    int b = bt / E2;
    int t = bt % E2;
    int c = idx % E1;
    return {b, t, c};
}

__host__ __device__ cuda::std::tuple<int, int, int, int> i2n(int idx, int E1, int E2, int E3) {
    int b = idx / (E1 * E2 * E3);
    int rest = idx % (E1 * E2 * E3);
    int nh_ = rest / (E2 * E3);
    rest = rest % (E2 * E3);
    int t = rest / E3;
    int hs = rest % E3;
    return {b, t, nh_, hs};
}

void encoder_forward(float* out,
                     const thrust::device_vector<int>& inpv, float* wte, float* wpe,
                     int B, int T, int C, int V) {
    cuda::std::mdspan<float, cuda::std::dextents<int, 3>> out_md(out, B, T, C);
    cuda::std::mdspan<float, cuda::std::dextents<int, 2>> wte_md(wte, V, C);
    cuda::std::mdspan<float, cuda::std::dextents<int, 2>> wpe_md(wpe, T, C);
    cuda::std::mdspan<const int, cuda::std::dextents<int, 2>> inp_md(thrust::raw_pointer_cast(inpv.data()), B, T);

    cudaCheck(cub::DeviceFor::Bulk(B * T * C, [=] __device__(int idx) {
      auto [b, t, c] = i2n(idx, C, T);
      out_md(b, t, c) = wte_md(inp_md(b, t), c) + wpe_md(t, c);
    }));
}

void encoder_backward(float* dwte, float* dwpe,
                    const float* dout, const int* inp,
                    int B, int T, int C) {
    const int N = B * T * C;
    const int block_size = 256;
    const int grid_size = CEIL_DIV(N, block_size);
    encoder_backward_kernel<<<grid_size, block_size>>>(dwte, dwpe, dout, inp, B, T, C);
    cudaCheck(cudaGetLastError());
}

void layernorm_forward(float* out, float* mean, float* rstd,
                       float* inp, float* weight, float* bias,
                       int B, int T, int C) {
    const int block_size = 512;
    const int N = B * T;
    const int grid_size = CEIL_DIV(N * 32, block_size);
    layernorm_forward_kernel3<<<grid_size, block_size>>>(out, mean, rstd, inp, weight, bias, N, C);
cudaDeviceSynchronize();
    cudaCheck(cudaGetLastError());
}

// uses cuBLAS
void matmul_forward_cublas(float* out,
                    float* inp, float* weight, float* bias,
                    int B, int T, int C, int OC) {
    assert(bias == NULL); // bias is not supported for this kernel
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasCheck(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, OC, B*T, C, &alpha, weight, C, inp, C, &beta, out, OC));
}

// uses cuBLASLt to fuse the bias and gelu. does not work with OC = 50257 (last layer)
// https://docs.nvidia.com/cuda/cublas/#cublasltmatmul
// https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuBLASLt/LtSgemm/sample_cublasLt_LtSgemm.cu
void matmul_forward_cublaslt(float* out,
                     float* inp, float* weight, float* bias,
                     int B, int T, int C, int OC) {
    int has_bias = (bias != NULL);

    // check bias alignment
    if(((uintptr_t)bias % 16) != 0) {
        printf("Bias pointer is not aligned (cuBLASLt requirement)!\n");
        exit(EXIT_FAILURE);
    }

    int returnedResults = 0;
    cublasLtMatmulDesc_t operationDesc;
    cublasLtMatmulPreference_t preference;
    cublasLtMatrixLayout_t weightLayout;
    cublasLtMatrixLayout_t inputLayout;
    cublasLtMatrixLayout_t outputLayout;
    cublasLtMatrixLayout_t biasLayout;
    cublasLtMatmulHeuristicResult_t heuristic;

    // create the operation descriptor
    cublasOperation_t opNoTranspose = CUBLAS_OP_N;
    cublasOperation_t opTranspose = CUBLAS_OP_T;
    cublasLtEpilogue_t epilogueBias = CUBLASLT_EPILOGUE_BIAS;
    cublasCheck(cublasLtMatmulDescCreate(&operationDesc, cublas_compute_type, CUDA_R_32F));
    cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opTranspose, sizeof(opTranspose)));
    cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opNoTranspose, sizeof(opNoTranspose)));
    cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogueBias, sizeof(epilogueBias)));
    cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));

    // define matrix layouts
    cublasCheck(cublasLtMatrixLayoutCreate(&weightLayout, CUDA_R_32F, C, OC, C));
    cublasCheck(cublasLtMatrixLayoutCreate(&inputLayout, CUDA_R_32F, C, B*T, C));
    cublasCheck(cublasLtMatrixLayoutCreate(&outputLayout, CUDA_R_32F, OC, B*T, OC));
    cublasCheck(cublasLtMatrixLayoutCreate(&biasLayout, CUDA_R_32F, OC, 1, OC));

    // create a preference handle with specified max workspace
    cublasCheck(cublasLtMatmulPreferenceCreate(&preference));
    cublasCheck(cublasLtMatmulPreferenceSetAttribute(preference,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &cublaslt_workspace_size, sizeof(cublaslt_workspace_size)));

    // find a suitable algorithm
    cublasCheck(cublasLtMatmulAlgoGetHeuristic(cublaslt_handle, operationDesc,
        weightLayout, inputLayout, outputLayout, outputLayout,
        preference, 1, &heuristic, &returnedResults));
    if (returnedResults == 0) {
        printf("No cuBLASLt algorithm: B: %d, T: %d, C: %d, OC: %d, bias: %d\n", B, T, C, OC, has_bias);
        exit(EXIT_FAILURE);
    }

    // call the matmul
    const float alpha = 1.0f, beta = 0.0f;
    cublasCheck(cublasLtMatmul(cublaslt_handle, operationDesc,
        &alpha, weight, weightLayout, inp, inputLayout, &beta,
        out, outputLayout, out, outputLayout, &heuristic.algo,
        cublaslt_workspace, cublaslt_workspace_size, 0));

    // cleanups
    cublasCheck(cublasLtMatmulPreferenceDestroy(preference));
    cublasCheck(cublasLtMatmulDescDestroy(operationDesc));
    cublasCheck(cublasLtMatrixLayoutDestroy(weightLayout));
    cublasCheck(cublasLtMatrixLayoutDestroy(inputLayout));
    cublasCheck(cublasLtMatrixLayoutDestroy(outputLayout));
    cublasCheck(cublasLtMatrixLayoutDestroy(biasLayout));
}

template <class T> struct streaming_accessor {
    using offset_policy = streaming_accessor;
    using element_type = T;
    using data_handle_type = const T *;
    using reference = const T;

    inline __host__ __device__ 
    reference access(data_handle_type p, size_t i) const {
        NV_IF_TARGET(NV_IS_DEVICE, (return __ldcs(p + i);), (return p[i];));
    }

    inline __host__ __device__ 
    data_handle_type offset(data_handle_type p, size_t i) const {
        return p + i;
    }
};

void attention_forward(float* out, float* vaccum, float* qkvr, float* preatt, float* att,
                        float* inp,
                        int B, int T, int C, int NH, int l) {
    const int block_size = 256;
    const int softmax_block_size = 256;

    // inp is (B, T, 3C) QKV
    // preatt, att are (B, NH, T, T)
    // output is (B, T, C)
    int HS = C / NH; // head size

    // permute and separate inp from (B, T, 3, NH, HS) to 3X (B, NH, T, HS)
    float *q, *k, *v;
    q = qkvr + 0 * B * T * C;
    k = qkvr + 1 * B * T * C;
    v = qkvr + 2 * B * T * C;

    constexpr auto dyn = cuda::std::dynamic_extent;
    using ext_t = cuda::std::extents<int, dyn, dyn, 3, dyn, dyn>;
    using mds_t = cuda::std::mdspan<const float, ext_t, cuda::std::layout_right, streaming_accessor<float>>;

    ext_t extents{B, T, NH, HS};
    mds_t inp_md{inp, extents};

    // Q[b][nh_][n][d_] = inp[b][n][0][nh_][d_]
    thrust::for_each(thrust::cuda::par_nosync, thrust::make_counting_iterator(0),
                     thrust::make_counting_iterator(B * NH * T * HS),
                     [=] __device__(int idx) {
                        auto [b, t, nh_, hs] = i2n(idx, NH, T, HS);
                        q[idx] = inp_md(b, t, 0, nh_, hs);
                        k[idx] = inp_md(b, t, 1, nh_, hs);
                        v[idx] = inp_md(b, t, 2, nh_, hs);
                     });

    //dump_tensor(q, std::to_string(l)+"_q", B * NH * T * HS);
    //dump_tensor(k, std::to_string(l)+"_k", B * NH * T * HS);
    //dump_tensor(v, std::to_string(l)+"_v", B * NH * T  *HS);
    // batched matrix multiply with cuBLAS
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasCheck(cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, T, T, HS, &alpha, k, HS, T * HS, q, HS, T * HS, &beta, preatt, T, T * T, B * NH));
    // dump_tensor(preatt, std::to_string(l)+"_preatt", B * NH * T  * T);
    // multiply all elements of preatt elementwise by scale
    float scale = 1.0 / sqrtf(HS);
    int grid_size = CEIL_DIV(B * NH * T * 32, softmax_block_size);
    softmax_forward_kernel5<<<grid_size, softmax_block_size>>>(att, scale, preatt, B * NH, T);
    // dump_tensor(att, std::to_string(l)+"_att", B * NH * T  * T);
    cudaCheck(cudaGetLastError());
    // new approach: first cuBLAS another batched matmul
    // y = att @ v # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
    cublasCheck(cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, HS, T, T, &alpha, v, HS, T * HS, att, T, T * T, &beta, vaccum, HS, T * HS, B * NH));
    //dump_tensor(vaccum, std::to_string(l)+"_vaccum", B *  T  * C);

    // now unpermute
    // y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
    auto map = thrust::make_transform_iterator(
        thrust::make_counting_iterator(0), [=] __host__ __device__(int idx) {
          auto [b, n, nh_, d_] = i2n(idx, NH, T, HS);
          return (b * NH * T * HS) + (n * NH * HS) + (nh_ * HS) + d_;
        });
    cub::CacheModifiedInputIterator<cub::LOAD_CS, float> vaccumcs(vaccum);
    thrust::scatter(thrust::cuda::par_nosync, vaccumcs, vaccumcs + B * T * C, map, out);
}

void residual_forward(float* out, float* inp1, float* inp2, int N) {
    cub::CacheModifiedInputIterator<cub::LOAD_CS, float> inp1cs(inp1);
    cub::CacheModifiedInputIterator<cub::LOAD_CS, float> inp2cs(inp2);
    thrust::transform(thrust::cuda::par_nosync, inp1cs, inp1cs + N, inp2cs, out, thrust::plus<float>());
}

void gelu_forward(float* out, const float* inp, int N) {
    thrust::transform(thrust::cuda::par_nosync, inp, inp + N, out, [] __device__(float xi) {
        float cube = 0.044715f * xi * xi * xi;
        return 0.5f * xi * (1.0f + tanhf(GELU_SCALING_FACTOR * (xi + cube)));
    });
}

void gelu_backward(float* dinp, const float* inp, const float* dout, const int N) {
    const int block_size = 128;
    const int grid_size = CEIL_DIV(N, block_size);
    gelu_backward_kernel<<<grid_size, block_size>>>(dinp, inp, dout, N);
    cudaCheck(cudaGetLastError());
}

void softmax_forward(float* out, float* inp, int N, int C) {
    int grid_size = N;
    const int block_size = 512;
    size_t shared_mem_size = 2 * block_size / 32 * sizeof(float);
    softmax_forward_kernel7<<<grid_size, block_size, shared_mem_size>>>(out, inp, N, C);
    cudaCheck(cudaGetLastError());
}

void matmul_backward(float* dinp, float* dweight, float* dbias,
                     float* dout, float* inp, float* weight,
                     int B, int T, int C, int OC) {
    float one = 1.0f;
    float zero = 0.0f;
    // backward to input, uses = in the backward pass (set the gradient)
    cublasCheck(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, C, B*T, OC, &one, weight, C, dout, OC, &zero, dinp, C));
    // backward to weight, uses += in the backward pass (accumulate the gradient)
    cublasCheck(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, C, OC, B*T, &one, inp, C, dout, OC, &one, dweight, C));
    // backward to bias, if given, does a +=
    if (dbias != NULL) {
        const int block_size = 512;
        const int grid_size = CEIL_DIV(OC * 32, block_size);
        matmul_backward_bias_kernel2<<<grid_size, block_size>>>(dbias, dout, B, T, OC);
        cudaCheck(cudaGetLastError());
    }
}

void layernorm_backward(float* dinp, float* dweight, float* dbias,
                        const float* dout, const float* inp, const  float* weight, const float* mean, const float* rstd,
                        int B, int T, int C) {
    const int block_size = 256;
    const int N = B * T;
    // one warp per token, so we need to divide by 32 here.
    const int grid_size = CEIL_DIV(N, block_size / 32);
    layernorm_backward_kernel<<<grid_size, block_size>>>(dinp, dweight, dbias, dout, inp, weight, mean, rstd, B, T, C);
    cudaCheck(cudaGetLastError());
}

// the sequence of transformations in this compound op is:
// inp (B,T,3C) -> qkvr (B,T,3C) -> preatt (B,NH,T,T) -> att (B,NH,T,T) -> vaccum (B,T,C) -> out (B,T,C)
void attention_backward(float* dinp, float* dqkvr, float* dpreatt, float* datt, float* dvaccum,
                        const float* dout,
                        const float* inp, const float* qkvr, const float* att,
                        int B, int T, int C, int NH) {
    const int block_size = 256;
    int HS = C / NH; // head size
    const float one = 1.0f;
    const float zero = 0.0f; // note beta = 1.0f so that we accumulate gradients (+=)
    // unpack convenience pointers into q, k, v
    const float *q, *k, *v;
    q = qkvr + 0 * B * T * C;
    k = qkvr + 1 * B * T * C;
    v = qkvr + 2 * B * T * C;
    float *dq, *dk, *dv;
    dq = dqkvr + 0 * B * T * C;
    dk = dqkvr + 1 * B * T * C;
    dv = dqkvr + 2 * B * T * C;
    // backward through the unpermute operation
    int num_blocks = CEIL_DIV(B * T * C, block_size);
    unpermute_kernel_backward<<<num_blocks, block_size>>>(dvaccum, dout, B, T, NH, HS);
    cudaCheck(cudaGetLastError());
    // backward into datt
    cublasCheck(cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, T, T, HS, &one, v, HS, T * HS, dvaccum, HS, T * HS, &zero, datt, T, T * T, B * NH));
    // backward into dv
    cublasCheck(cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, HS, T, T, &one, dvaccum, HS, T * HS, att, T, T * T, &zero, dv, HS, T * HS, B * NH));
    // backward into preatt
    int hs = C / NH; // head size
    float scale = 1.0f / sqrtf(hs);
    softmax_autoregressive_backward_kernel<<<dim3(T / 4, B * NH), 256>>>(dpreatt, datt, att, B, T, C, scale);
    cudaCheck(cudaGetLastError());
    // backward into q
    cublasCheck(cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, HS, T, T, &one, k, HS, T * HS, dpreatt, T, T * T, &zero, dq, HS, T * HS, B * NH));
    // backward into k
    cublasCheck(cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, HS, T, T, &one, q, HS, T * HS, dpreatt, T, T * T, &zero, dk, HS, T * HS, B * NH));
    // backward into inp
    num_blocks = CEIL_DIV(B * NH * T * HS, block_size);
    permute_kernel_backward<<<num_blocks, block_size>>>(dinp, dq, dk, dv, B, T, NH, HS);
    cudaCheck(cudaGetLastError());
}

// replaces logits with logit gradients
void fused_classifier3(float* logits, float* losses,
                      const float* dlosses, const int* targets,
                      int B, int T, int V, int P) {
    const int block_size = 1024;
    const int N = B * T;
    const int grid_size = N;
    fused_classifier_kernel3<<<grid_size, block_size>>>(logits, losses, NULL, dlosses, targets, B, T, V, P);
    cudaCheck(cudaGetLastError());
}

// ----------------------------------------------------------------------------
// GPT-2 model definition

typedef struct {
    int max_seq_len; // max sequence length, e.g. 1024
    int vocab_size; // vocab size, e.g. 50257
    int num_layers; // number of layers, e.g. 12
    int num_heads; // number of heads in attention, e.g. 12
    int channels; // number of channels, e.g. 768
} GPT2Config;

// the parameters of the model
#define NUM_PARAMETER_TENSORS 16
typedef struct {
    float* wte; // (V, C)
    float* wpe; // (maxT, C)
    float* ln1w; // (L, C)
    float* ln1b; // (L, C)
    float* qkvw; // (L, 3*C, C)
    float* qkvb; // (L, 3*C)
    float* attprojw; // (L, C, C)
    float* attprojb; // (L, C)
    float* ln2w; // (L, C)
    float* ln2b; // (L, C)
    float* fcw; // (L, 4*C, C)
    float* fcb; // (L, 4*C)
    float* fcprojw; // (L, C, 4*C)
    float* fcprojb; // (L, C)
    float* lnfw; // (C)
    float* lnfb; // (C)
} ParameterTensors;

void fill_in_parameter_sizes(size_t* param_sizes, GPT2Config config) {
    int V = config.vocab_size;
    int C = config.channels;
    int maxT = config.max_seq_len;
    int L = config.num_layers;
    param_sizes[0] = V * C; // wte
    param_sizes[1] = maxT * C; // wpe
    param_sizes[2] = L * C; // ln1w
    param_sizes[3] = L * C; // ln1b
    param_sizes[4] = L * (3 * C) * C; // qkvw
    param_sizes[5] = L * (3 * C); // qkvb
    param_sizes[6] = L * C * C; // attprojw
    param_sizes[7] = L * C; // attprojb
    param_sizes[8] = L * C; // ln2w
    param_sizes[9] = L * C; // ln2b
    param_sizes[10] = L * (4 * C) * C; // fcw
    param_sizes[11] = L * (4 * C); // fcb
    param_sizes[12] = L * C * (4 * C); // fcprojw
    param_sizes[13] = L * C; // fcprojb
    param_sizes[14] = C; // lnfw
    param_sizes[15] = C; // lnfb
}

// allocate memory for the parameters and point the individual tensors to the right places
thrust::device_vector<float> malloc_and_point_parameters(ParameterTensors* params, size_t* param_sizes) {
    // calculate the number of parameters
    size_t num_parameters = 0;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        num_parameters += param_sizes[i];
    }
    thrust::device_vector<float> params_memory(num_parameters);
    // assign all the tensors their place in the array
    float** ptrs[] = {
        &params->wte, &params->wpe, &params->ln1w, &params->ln1b, &params->qkvw, &params->qkvb,
        &params->attprojw, &params->attprojb, &params->ln2w, &params->ln2b, &params->fcw, &params->fcb,
        &params->fcprojw, &params->fcprojb, &params->lnfw, &params->lnfb
    };
    float* params_memory_iterator = thrust::raw_pointer_cast(params_memory.data());
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        *(ptrs[i]) = params_memory_iterator;
        params_memory_iterator += param_sizes[i];
    }
    return params_memory;
}

#define NUM_ACTIVATION_TENSORS 25
typedef struct {
    float* encoded; // (B, T, C)
    float* ln1; // (L, B, T, C)
    float* ln1_mean; // (L, B, T)
    float* ln1_rstd; // (L, B, T)
    float* qkv; // (L, B, T, 3*C)
    float* atty; // (L, B, T, C)
    float* preatt; // (L, B, NH, T, T)
    float* att; // (L, B, NH, T, T)
    float* attproj; // (L, B, T, C)
    float* residual2; // (L, B, T, C)
    float* ln2; // (L, B, T, C)
    float* ln2_mean; // (L, B, T)
    float* ln2_rstd; // (L, B, T)
    float* fch; // (L, B, T, 4*C)
    float* fch_gelu; // (L, B, T, 4*C)
    float* fcproj; // (L, B, T, C)
    float* residual3; // (L, B, T, C)
    float* lnf; // (B, T, C)
    float* lnf_mean; // (B, T)
    float* lnf_rstd; // (B, T)
    // if we have targets, this will be the logit _gradients_.
    float* logits; // (B, T, V)
    float* probs; // (B, T, V)
    float* losses; // (B, T)
    // adding these two compared to the CPU .c code, needed for attention kernel as buffers
    float* qkvr; // (L, B, T, 3*C)
    float* v_accum; // (L, B, T, C)
} ActivationTensors;

void fill_in_activation_sizes(size_t* act_sizes, int B, int T, GPT2Config config) {
    int V = config.vocab_size;
    int L = config.num_layers;
    int NH = config.num_heads;
    int C = config.channels;
    act_sizes[0] = B * T * C; // encoded
    act_sizes[1] = L * B * T * C; // ln1
    act_sizes[2] = L * B * T; // ln1_mean
    act_sizes[3] = L * B * T; // ln1_rstd
    act_sizes[4] = L * B * T * 3*C; // qkv
    act_sizes[5] = L * B * T * C; // atty
    act_sizes[6] = B * NH * T * T; // preatt
    act_sizes[7] = L * B * NH * T * T; // att
    act_sizes[8] = L * B * T * C; // attproj
    act_sizes[9] = L * B * T * C; // residual2
    act_sizes[10] = L * B * T * C; // ln2
    act_sizes[11] = L * B * T; // ln2_mean
    act_sizes[12] = L * B * T; // ln2_rstd
    act_sizes[13] = L * B * T * 4*C; // fch
    act_sizes[14] = L * B * T * 4*C; // fch_gelu
    act_sizes[15] = L * B * T * C; // fcproj
    act_sizes[16] = L * B * T * C; // residual3
    act_sizes[17] = B * T * C; // lnf
    act_sizes[18] = B * T; // lnf_mean
    act_sizes[19] = B * T; // lnf_rstd
    act_sizes[20] = B * T * V; // logits
    act_sizes[21] = B * T * V; // probs
    act_sizes[22] = B * T; // losses
    act_sizes[23] = L * B * T * 3*C; // qkvr
    act_sizes[24] = B * T * C; // v_accum
}

thrust::device_vector<float> malloc_and_point_activations(ActivationTensors* acts, size_t* act_sizes) {
    size_t num_activations = 0;
    for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
        num_activations += act_sizes[i];
    }
    thrust::device_vector<float> acts_memory(num_activations);
    float** ptrs[] = {
        &acts->encoded, &acts->ln1, &acts->ln1_mean, &acts->ln1_rstd, &acts->qkv, &acts->atty,
        &acts->preatt, &acts->att, &acts->attproj, &acts->residual2, &acts->ln2, &acts->ln2_mean,
        &acts->ln2_rstd, &acts->fch, &acts->fch_gelu, &acts->fcproj, &acts->residual3, &acts->lnf,
        &acts->lnf_mean, &acts->lnf_rstd, &acts->logits, &acts->probs, &acts->losses,
        &acts->qkvr, &acts->v_accum
    };
    float* acts_memory_iterator = thrust::raw_pointer_cast(acts_memory.data());
    for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
        *(ptrs[i]) = acts_memory_iterator;
        acts_memory_iterator += act_sizes[i];
    }
    return acts_memory;
}

struct GPT2 {
    GPT2Config config;
    // the weights of the model, and their sizes
    ParameterTensors params;
    size_t param_sizes[NUM_PARAMETER_TENSORS];
    thrust::device_vector<float> params_memory;
    size_t num_parameters;
    // gradients of the weights
    ParameterTensors grads;
    thrust::device_vector<float> grads_memory;
    thrust::device_vector<float> m_memory;
    thrust::device_vector<float> v_memory;
    // the activations of the model, and their sizes
    ActivationTensors acts;
    size_t act_sizes[NUM_ACTIVATION_TENSORS];
    thrust::device_vector<float> acts_memory;
    size_t num_activations;
    // gradients of the activations
    ActivationTensors grads_acts;
    size_t num_grad_acts;
    thrust::device_vector<float> grads_acts_memory;
    // other run state configuration
    int batch_size; // the batch size (B) of current forward pass
    int seq_len; // the sequence length (T) of current forward pass
    thrust::device_vector<int> inputs; // the input tokens for the current forward pass
    thrust::device_vector<int> targets; // the target tokens for the current forward pass
    float mean_loss; // after a forward pass with targets, will be populated with the mean loss
    pinned_vector<float> cpu_losses; // CPU buffer to copy the losses to, allocated with cudaMallocHost
};


void gpt2_build_from_checkpoint(GPT2 &model, const char* checkpoint_path) {

    // read in model from a checkpoint file
    FILE *model_file = fopenCheck(checkpoint_path, "rb");
    int model_header[256];
    fread(model_header, sizeof(int), 256, model_file);
    if (model_header[0] != 20240326) { printf("Bad magic model file"); exit(1); }
    if (model_header[1] != 1) { printf("Bad version in model file"); exit(1); }

    // read in hyperparameters
    int maxT, V, L, NH, C;
    model.config.max_seq_len = maxT = model_header[2];
    model.config.vocab_size = V = model_header[3];
    model.config.num_layers = L = model_header[4];
    model.config.num_heads = NH = model_header[5];
    model.config.channels = C = model_header[6];
    printf("[GPT-2]\n");
    printf("max_seq_len: %d\n", maxT);
    printf("vocab_size: %d\n", V);
    printf("num_layers: %d\n", L);
    printf("num_heads: %d\n", NH);
    printf("channels: %d\n", C);

    // allocate space for all the parameters and read them in
    fill_in_parameter_sizes(model.param_sizes, model.config);

    // count the number of parameters
    size_t num_parameters = 0;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        num_parameters += model.param_sizes[i];
    }
    printf("num_parameters: %zu\n", num_parameters);
    model.num_parameters = num_parameters;

    // create memory for model parameters on the device
    model.params_memory = malloc_and_point_parameters(&model.params, model.param_sizes);
    printf("allocated %d MiB for model parameters\n", (int)round(num_parameters * sizeof(float) / (1024 * 1024)));

    // read in all the parameters from file and copy them to device
    thrust::host_vector<float> params_memory_cpu(num_parameters);
    fread(thrust::raw_pointer_cast(params_memory_cpu.data()), sizeof(float), num_parameters, model_file);
    model.params_memory = params_memory_cpu;
    fclose(model_file);

    // other inits
    model.batch_size = 0;
    model.seq_len = 0;
    model.mean_loss = -1.0f; // -1.0f will designate no loss
}

void gpt2_forward(GPT2 &model, int* inputs, int* targets, int B, int T) {
    // targets are optional and could be NULL

    // ensure the model was initialized or error out
    if (model.params_memory.empty()) {
        printf("Error: model was not initialized properly.\n");
        exit(EXIT_FAILURE);
    }

    // convenience parameters
    int V = model.config.vocab_size;
    int L = model.config.num_layers;
    int NH = model.config.num_heads;
    int C = model.config.channels;
    // validate inputs, all indices must be in the range [0, V)
    for(int i = 0; i < B * T; i++) {
        assert(0 <= inputs[i] && inputs[i] < V);
        if (targets != NULL) {
            assert(0 <= targets[i] && targets[i] < V);
        }
    }

    // allocate space for all the activations if needed (done here, lazily)
    if(model.acts_memory.empty()) {
        // record the current B,T as well
        model.batch_size = B;
        model.seq_len = T;
        // and now allocate the space
        fill_in_activation_sizes(model.act_sizes, B, T, model.config);
        size_t num_activations = 0;
        for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
            num_activations += model.act_sizes[i];
        }
        printf("num_activations: %zu\n", num_activations);
        model.num_activations = num_activations;
        model.acts_memory = malloc_and_point_activations(&model.acts, model.act_sizes);
        printf("allocated %d MiB for activations\n", (int)round(num_activations * sizeof(float) / (1024 * 1024)));
        // also create memory for caching inputs and targets
        model.inputs.resize(B * T);
        model.targets.resize(B * T);
        model.cpu_losses.resize(B * T);
    } else {
        // validate B,T is consistent with how we've allocated the memory before
        // in principle we could get more clever here in the future, for now this is safest
        if (B != model.batch_size || T != model.seq_len) {
            printf("Model: B=%d T=%d, Desired: B=%d T=%d\n", model.batch_size, model.seq_len, B, T);
            exit(EXIT_FAILURE);
        }
    }

    // copy inputs/targets to the model
    thrust::copy_n(inputs, B * T, model.inputs.begin());
    if (targets != NULL) {
        thrust::copy_n(targets, B * T, model.targets.begin());
    }

    // forward pass
    ParameterTensors params = model.params; // for brevity
    ActivationTensors acts = model.acts;
    float* residual;
    encoder_forward(acts.encoded, model.inputs, params.wte, params.wpe, B, T, C, V); // encoding goes into residual[0]
    //dump_tensor(acts.encoded, "acts.encoded", B * T * C);
    //dump_tensor(thrust::raw_pointer_cast(model.inputs.data()), "model.inputs", B * T);
    //dump_tensor(params.wte, "params.wte", V * C);
    //dump_tensor(params.wpe, "params.wpe", T * C);

    for (int l = 0; l < L; l++) {

        residual = l == 0 ? acts.encoded : acts.residual3 + (l-1) * B * T * C;
        // get the pointers of the weights for this layer
        float* l_ln1w = params.ln1w + l * C;
        float* l_ln1b = params.ln1b + l * C;
        float* l_qkvw = params.qkvw + l * 3*C * C;
        float* l_qkvb = params.qkvb + l * 3*C;
        float* l_attprojw = params.attprojw + l * C * C;
        float* l_attprojb = params.attprojb + l * C;
        float* l_ln2w = params.ln2w + l * C;
        float* l_ln2b = params.ln2b + l * C;
        float* l_fcw = params.fcw + l * 4*C * C;
        float* l_fcb = params.fcb + l * 4*C;
        float* l_fcprojw = params.fcprojw + l * C * 4*C;
        float* l_fcprojb = params.fcprojb + l * C;

        // get the pointers of the activations for this layer
        float* l_ln1 = acts.ln1 + l * B * T * C;
        float* l_ln1_mean = acts.ln1_mean + l * B * T;
        float* l_ln1_rstd = acts.ln1_rstd + l * B * T;
        float* l_qkv = acts.qkv + l * B * T * 3*C;
        float* l_qkvr = acts.qkvr + l * B * T * 3*C;
        float* l_atty = acts.atty + l * B * T * C;
        float* l_att = acts.att + l * B * NH * T * T;
        float* l_attproj = acts.attproj + l * B * T * C;
        float* l_residual2 = acts.residual2 + l * B * T * C;
        float* l_ln2 = acts.ln2 + l * B * T * C;
        float* l_ln2_mean = acts.ln2_mean + l * B * T;
        float* l_ln2_rstd = acts.ln2_rstd + l * B * T;
        float* l_fch = acts.fch + l * B * T * 4*C;
        float* l_fch_gelu = acts.fch_gelu + l * B * T * 4*C;
        float* l_fcproj = acts.fcproj + l * B * T * C;
        float* l_residual3 = acts.residual3 + l * B * T * C;
        // these are only needed as scratchpads for the forward pass, but
        // need not be stored for backward
        float* l_preatt = acts.preatt;
        float* l_v_accum = acts.v_accum;

        // now do the forward pass
     //dump_tensor(residual, std::to_string(l)+"_residual", B * T * C);
        layernorm_forward(l_ln1, l_ln1_mean, l_ln1_rstd, residual, l_ln1w, l_ln1b, B, T, C);
     //dump_tensor(l_ln1, std::to_string(l)+"_l_ln1", B * T * C);
    // dump_tensor(l_ln1_mean, std::to_string(l)+"_l_ln1_mean", B * T);
    // dump_tensor(l_ln1_rstd, std::to_string(l)+"_l_ln1_rstd", B * T);
    //dump_tensor(l_ln1w, std::to_string(l)+"_l_ln1w", C);
    //dump_tensor(l_ln1b, std::to_string(l)+"_l_ln1b", C);
        matmul_forward_cublaslt(l_qkv, l_ln1, l_qkvw, l_qkvb, B, T, C, 3*C);
    //dump_tensor(l_qkv, std::to_string(l)+"_l_qkv", B * T * 3 * C);
    // dump_tensor(l_qkvw, std::to_string(l)+"_l_qkvw", 3 * C * C);
    // dump_tensor(l_qkvb, std::to_string(l)+"_l_qkvb", 3 * C);
        attention_forward(l_atty, l_v_accum, l_qkvr, l_preatt, l_att, l_qkv, B, T, C, NH, l);
    //dump_tensor(l_atty, std::to_string(l)+"_l_atty", B * T * C);
    //dump_tensor(l_v_accum, std::to_string(l)+"_l_v_accum", B * T * C);
    //dump_tensor(l_qkvr, std::to_string(l)+"_l_qkvr", B * T * 3 * C);
    //dump_tensor(l_preatt, std::to_string(l)+"_l_preatt", B * NH * T * T);
    //dump_tensor(l_att, std::to_string(l)+"_l_att", B * NH * T * T);
        matmul_forward_cublaslt(l_attproj, l_atty, l_attprojw, l_attprojb, B, T, C, C);
    //dump_tensor(l_attproj, std::to_string(l)+"_l_attproj", B * T * C);
        residual_forward(l_residual2, residual, l_attproj, B*T*C);
    //dump_tensor(l_residual2, std::to_string(l)+"_l_residual2", B * T * C);
        layernorm_forward(l_ln2, l_ln2_mean, l_ln2_rstd, l_residual2, l_ln2w, l_ln2b, B, T, C);
    //dump_tensor(l_ln2, std::to_string(l)+"_l_ln2", B * T * C);
        matmul_forward_cublaslt(l_fch, l_ln2, l_fcw, l_fcb, B, T, C, 4*C);
    //dump_tensor(l_fch, std::to_string(l)+"_l_fch", B * T * 4 * C);
        gelu_forward(l_fch_gelu, l_fch, B*T*4*C);
    //dump_tensor(l_fch_gelu, std::to_string(l)+"_l_fch_gelu", B * T * 4 * C);
        matmul_forward_cublaslt(l_fcproj, l_fch_gelu, l_fcprojw, l_fcprojb, B, T, 4*C, C);
    //dump_tensor(l_fcproj, std::to_string(l)+"_l_fcproj", B * T * C);
    //dump_tensor(l_residual2, std::to_string(l)+"_l_residual2", B * T * C);
        residual_forward(l_residual3, l_residual2, l_fcproj, B*T*C);
    //dump_tensor(l_residual3, std::to_string(l)+"_l_residual3", B * T * C);
    }

    residual = acts.residual3 + (L-1) * B * T * C; // last residual is in residual3
    layernorm_forward(acts.lnf, acts.lnf_mean, acts.lnf_rstd, residual, params.lnfw, params.lnfb, B, T, C);
    matmul_forward_cublas(acts.logits, acts.lnf, params.wte, NULL, B, T, C, V);

    // also forward the cross-entropy loss function if we have the targets
    if (targets != NULL) {
        // fused classifier: does the forward pass and first part of the backward pass
        // we're passing dlosses = NULL, which will default them to 1.0f/(B*T), i.e. uniform loss
        // dump_tensor(acts.logits, "logits_i", B * T * V);
        fused_classifier3(acts.logits, acts.losses, NULL, thrust::raw_pointer_cast(model.targets.data()), B, T, V, V); 
        // dump_tensor(acts.logits, "logits", B * T * V);
        // dump_tensor(acts.losses, "losses", B * T);
        //exit(1);
        // for convenience also evaluate the mean loss (TODO re-think this compute+sync point)
        // move the (B,T) losses to CPU
        thrust::copy_n(thrust::device_pointer_cast(acts.losses), B * T, model.cpu_losses.begin());
        float mean_loss = 0.0f;
        for (int i=0; i<B*T; i++) { mean_loss += model.cpu_losses[i]; }
        mean_loss /= B*T;
        model.mean_loss = mean_loss;
    } else {
        // if we don't have targets, we don't have loss
        softmax_forward(acts.probs, acts.logits, B*T, V);
        model.mean_loss = -1.0f;
    }
}

void gpt2_zero_grad(GPT2 &model) {
    thrust::fill(thrust::cuda::par_nosync, model.grads_acts_memory.begin(), model.grads_acts_memory.end(), 0.0f);
    thrust::fill(thrust::cuda::par_nosync,model.grads_memory.begin(), model.grads_memory.end(), 0.0f);
}

void gpt2_backward(GPT2 &model) {

    // double check we forwarded previously, with targets
    if (model.mean_loss == -1.0f) {
        printf("Error: must forward with targets before backward\n");
        exit(EXIT_FAILURE);
    }

    // lazily allocate the memory for gradients of the weights and activations, if needed
    if (model.grads_memory.empty()) {
        // allocate buffers for weight gradients
        model.grads_memory = malloc_and_point_parameters(&model.grads, model.param_sizes);
        printf("allocated %d MiB for parameter gradients\n", (int)round(model.num_parameters * sizeof(float) / (1024 * 1024)));
        // we're going to be clever for the activations backward pass. we don't need to exactly
        // mirror the forward pass acrtivations and we will save memory.
        size_t bw_act_sizes[NUM_ACTIVATION_TENSORS];
        GPT2Config cfg = model.config;
        cfg.num_layers = 1; // copy the configuration but override number of layers to 1
        fill_in_activation_sizes(bw_act_sizes, model.batch_size, model.seq_len, cfg);
        // on top of that, some buffers are not needed at all, set their sizes to zero
        bw_act_sizes[0] = 0; // encoded
        bw_act_sizes[2] = 0; // ln1_mean
        bw_act_sizes[3] = 0; // ln1_rstd
        bw_act_sizes[8] = 0; // attproj
        bw_act_sizes[9] = 0; // residual2
        bw_act_sizes[11] = 0; // ln2_mean
        bw_act_sizes[12] = 0; // ln2_rstd
        bw_act_sizes[18] = 0; // lnf_mean
        bw_act_sizes[19] = 0; // lnf_rstd
        bw_act_sizes[21] = 0; // probs
        // count up and allocate the space
        model.grads_acts_memory = malloc_and_point_activations(&model.grads_acts, bw_act_sizes);
        model.num_grad_acts = 0;
        for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
            model.num_grad_acts += bw_act_sizes[i];
        }
        printf("allocated %d MiB for activation gradients\n", (int)round(model.num_grad_acts * sizeof(float) / (1024 * 1024)));
        // init gradients of parameters and activations to zero
        gpt2_zero_grad(model);
    }

    // convenience shortcuts
    int B = model.batch_size;
    int T = model.seq_len;
    int V = model.config.vocab_size;
    int L = model.config.num_layers;
    int NH = model.config.num_heads;
    int C = model.config.channels;

    // backward pass: go in the reverse order of the forward pass, and call backward() functions
    ParameterTensors params = model.params; // for brevity
    ParameterTensors grads = model.grads;
    ActivationTensors acts = model.acts;
    ActivationTensors grads_acts = model.grads_acts;

    // we kick off the chain rule by filling in dlosses with 1.0f/(B*T)
    // this was done in the fused classifier kernel as last step of forward pass
    // technically that is a small, inline backward() pass of calculating
    // total, final loss as the mean over all losses over all (B,T) positions in the batch
    // next: backward the classifier matmul
    matmul_backward(grads_acts.lnf, grads.wte, NULL, acts.logits, acts.lnf, params.wte, B, T, C, V);
    //dump_tensor(grads_acts.lnf, "grad_acts.lnf", B * T * C);
    //dump_tensor(grads.wte, "grads.wte", V * C);
    //dump_tensor(acts.lnf, "acts.lnf", B * T * C);
    //dump_tensor(params.wte, "params.wte", V * C);
    // backward the final layernorm
    float* residual = acts.residual3 + (L-1) * B * T * C; // last residual is in residual3
    float* dresidual = grads_acts.residual3; // the main buffer holding the gradient in the backward pass
    //        dump_tensor(acts.lnf_mean, "acts.lnf_mean", B * T);
    //        dump_tensor(acts.lnf_rstd, "acts.lnf_rstd", B * T);
    //        dump_tensor(params.lnfw, "params.lnfw", C);
    //        dump_tensor(residual, "residual", B * T * C);
    //        dump_tensor(dresidual, "dresidual_i", B * T * C);
    layernorm_backward(dresidual, grads.lnfw, grads.lnfb, grads_acts.lnf, residual, params.lnfw, acts.lnf_mean, acts.lnf_rstd, B, T, C);
    //dump_tensor(thrust::raw_pointer_cast(model.grads_memory.data()), "grads", model.num_parameters);
    //        dump_tensor(dresidual, "dresidual_ln", B * T * C);
    // now backward all the layers
    for (int l = L-1; l >= 0; l--) {
        residual = l == 0 ? acts.encoded : acts.residual3 + (l-1) * B * T * C;

        // get the pointers of the weights for this layer
        float* l_ln1w = params.ln1w + l * C;
        float* l_qkvw = params.qkvw + l * 3*C * C;
        float* l_attprojw = params.attprojw + l * C * C;
        float* l_ln2w = params.ln2w + l * C;
        float* l_fcw = params.fcw + l * 4*C * C;
        float* l_fcprojw = params.fcprojw + l * C * 4*C;
        // get the pointers of the gradients of the weights for this layer
        float* dl_ln1w = grads.ln1w + l * C;
        float* dl_ln1b = grads.ln1b + l * C;
        float* dl_qkvw = grads.qkvw + l * 3*C * C;
        float* dl_qkvb = grads.qkvb + l * 3*C;
        float* dl_attprojw = grads.attprojw + l * C * C;
        float* dl_attprojb = grads.attprojb + l * C;
        float* dl_ln2w = grads.ln2w + l * C;
        float* dl_ln2b = grads.ln2b + l * C;
        float* dl_fcw = grads.fcw + l * 4*C * C;
        float* dl_fcb = grads.fcb + l * 4*C;
        float* dl_fcprojw = grads.fcprojw + l * C * 4*C;
        float* dl_fcprojb = grads.fcprojb + l * C;
        // get the pointers of the activations for this layer
        float* l_ln1 = acts.ln1 + l * B * T * C;
        float* l_ln1_mean = acts.ln1_mean + l * B * T;
        float* l_ln1_rstd = acts.ln1_rstd + l * B * T;
        float* l_qkv = acts.qkv + l * B * T * 3*C;
        float* l_qkvr = acts.qkvr + l * B * T * 3*C;
        float* l_atty = acts.atty + l * B * T * C;
        float* l_att = acts.att + l * B * NH * T * T;
        float* l_residual2 = acts.residual2 + l * B * T * C;
        float* l_ln2 = acts.ln2 + l * B * T * C;
        float* l_ln2_mean = acts.ln2_mean + l * B * T;
        float* l_ln2_rstd = acts.ln2_rstd + l * B * T;
        float* l_fch = acts.fch + l * B * T * 4*C;
        float* l_fch_gelu = acts.fch_gelu + l * B * T * 4*C;
        // get the pointers of the gradients of the activations for this layer
        // notice that there is no l *, because we just have a single copy, and keep
        // re-using this memory in every Transformer block as we calculate backward pass
        float* dl_ln1 = grads_acts.ln1;
        float* dl_qkv = grads_acts.qkv;
        float* dl_qkvr = grads_acts.qkvr;
        float* dl_atty = grads_acts.atty;
        float* dl_preatt = grads_acts.preatt;
        float* dl_att = grads_acts.att;
        float* dl_v_accum = grads_acts.v_accum;
        float* dl_ln2 = grads_acts.ln2;
        float* dl_fch = grads_acts.fch;
        float* dl_fch_gelu = grads_acts.fch_gelu;

        // backprop this layer
        //dump_tensor(dresidual, std::to_string(l)+"_dresidual", B *T * C);
        //dump_tensor(l_fch_gelu, std::to_string(l)+"_l_fch_gelu", B *T* 4 *C);
        //dump_tensor(l_fcprojw, std::to_string(l)+"_l_fcprojw", C * 4 * C);
        matmul_backward(dl_fch_gelu, dl_fcprojw, dl_fcprojb, dresidual, l_fch_gelu, l_fcprojw, B, T, 4*C, C);
        //dump_tensor(dl_fch_gelu, std::to_string(l)+"_dl_fch_gelu", B * T * 4 *C);
        //dump_tensor(dl_fcprojw, std::to_string(l)+"_dl_fcprojw",  C * 4 *C);
        //dump_tensor(dl_fcprojb, std::to_string(l)+"_dl_fcprojb", C);

        //dump_tensor(l_fch, std::to_string(l)+"_l_fch", B *T *4 * C);
        gelu_backward(dl_fch, l_fch, dl_fch_gelu, B*T*4*C);
        //dump_tensor(dl_fch, std::to_string(l)+"_dl_fch", B *T *4 *C);

        matmul_backward(dl_ln2, dl_fcw, dl_fcb, dl_fch, l_ln2, l_fcw, B, T, C, 4*C);
        // layernorm backward does += to the dresidual, so it correctly accumulates grad from the MLP block above
        
        //dump_tensor(dresidual, std::to_string(l)+"_dresidual_i", B*T*C);
        //dump_tensor(dl_ln2, std::to_string(l)+"_dl_ln2", B *T *C);
        //dump_tensor(l_residual2, std::to_string(l)+"_l_residual2", B *T *C);
        //dump_tensor(l_ln2w, std::to_string(l)+"_l_ln2w", C);
        //dump_tensor(l_ln2_mean, std::to_string(l)+"_l_ln2_mean", B *T );
        //dump_tensor(l_ln2_rstd, std::to_string(l)+"_l_ln2_rstd", B *T );
        layernorm_backward(dresidual, dl_ln2w, dl_ln2b, dl_ln2, l_residual2, l_ln2w, l_ln2_mean, l_ln2_rstd, B, T, C);
        //dump_tensor(dresidual, std::to_string(l)+"_dresidual", B*T*C);
        //dump_tensor(dl_ln2w, std::to_string(l)+"_dl_ln2w", C );
        //dump_tensor(dl_ln2b, std::to_string(l)+"_dl_ln2b", C );
        matmul_backward(dl_atty, dl_attprojw, dl_attprojb, dresidual, l_atty, l_attprojw, B, T, C, C);
        //dump_tensor(dl_ln2w, std::to_string(l)+"_dl_ln2w", C );
        //dump_tensor(dl_ln2b, std::to_string(l)+"_dl_ln2b", C );

        //dump_tensor(dl_atty, std::to_string(l)+"_dl_atty", B * T * C);
        //dump_tensor(l_qkv, std::to_string(l)+"_l_qkv", B * T * 3 * C);
        //dump_tensor(l_qkvr, std::to_string(l)+"_l_qkvr", B * T * 3 * C);
        //dump_tensor(l_att, std::to_string(l)+"_l_att", B * NH * T * T);
        attention_backward(dl_qkv, dl_qkvr, dl_preatt, dl_att, dl_v_accum, dl_atty, l_qkv, l_qkvr, l_att, B, T, C, NH);
        // dump_tensor(dl_qkv, std::to_string(l)+"_dl_qkv", B * T * 3 * C);
        // dump_tensor(dl_qkvr, std::to_string(l)+"_dl_qkvr", B * T * 3 * C);
        // dump_tensor(dl_preatt, std::to_string(l)+"_dl_preatt", B * NH * T * T);
        // dump_tensor(dl_att, std::to_string(l)+"_dl_att", B * NH * T * T);
        // dump_tensor(dl_v_accum, std::to_string(l)+"_dl_v_accum", B * T * C);
        matmul_backward(dl_ln1, dl_qkvw, dl_qkvb, dl_qkv, l_ln1, l_qkvw, B, T, C, 3*C);
        // layernorm backward does += to dresidual, so it correctly accumulates gradient for the Attention block above
        layernorm_backward(dresidual, dl_ln1w, dl_ln1b, dl_ln1, residual, l_ln1w, l_ln1_mean, l_ln1_rstd, B, T, C);
    }
    //dump_tensor(dresidual, "dresidual", B * T * C);
    //dump_tensor(grads.wte, "gwte_i", V * C);
    //dump_tensor(grads.wpe, "gwpe_i", T * C);
    encoder_backward(grads.wte, grads.wpe, dresidual, thrust::raw_pointer_cast(model.inputs.data()), B, T, C);
    //dump_tensor(grads.wte, "gwte", V * C);
    //dump_tensor(grads.wpe, "gwpe", T * C);
    //exit(0);
}

void gpt2_update(GPT2 *model, float learning_rate, float beta1, float beta2, float eps, float weight_decay, int t) {
    // reference: https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html

    // lazily allocate the memory for m_memory and v_memory
    if (model->m_memory.empty()) {
        model->m_memory = thrust::device_vector<float>(model->num_parameters, 0.0f);
        model->v_memory = thrust::device_vector<float>(model->num_parameters, 0.0f);
        printf("allocated %d MiB for AdamW optimizer state m\n", (int)round(model->num_parameters * sizeof(float) / (1024 * 1024)));
        printf("allocated %d MiB for AdamW optimizer state v\n", (int)round(model->num_parameters * sizeof(float) / (1024 * 1024)));
    }

    int block_size = 512;
    int num_blocks = CEIL_DIV(model->num_parameters, block_size);
    float beta1_correction = 1.0f - powf(beta1, t);
    float beta2_correction = 1.0f - powf(beta2, t);
    adamw_kernel2<<<num_blocks, block_size>>>(
        thrust::raw_pointer_cast(model->params_memory.data()), 
        thrust::raw_pointer_cast(model->grads_memory.data()), 
        thrust::raw_pointer_cast(model->m_memory.data()), 
        thrust::raw_pointer_cast(model->v_memory.data()),
        model->num_parameters,
        learning_rate, beta1, beta2, beta1_correction, beta2_correction, eps, weight_decay);
    cudaCheck(cudaGetLastError());
}

#ifndef TESTING
// if we are TESTING (see test_gpt2.cu), we'll skip the int main below

// ----------------------------------------------------------------------------
// data loader lite: returns random batches of data from a file of integers

struct DataLoader {
    // hyperparameters
    int B;
    int T;
    // input handling and its state
    FILE* tokens_file;
    long file_size;
    long current_position;
    // output memory
    pinned_vector<int> batch;
    int* inputs() {
        return thrust::raw_pointer_cast(batch.data());
    }
    int* targets() {
        // targets are shifted by one
        return inputs() + 1;
    }
    // convenience variables
    int num_batches;
};

void dataloader_init(DataLoader *loader, const char* filename, int B, int T) {
    loader->B = B;
    loader->T = T;

    // open the input file for reading
    loader->tokens_file = fopenCheck(filename, "rb");

    // determine the file size
    fseek(loader->tokens_file, 0, SEEK_END);
    loader->file_size = ftell(loader->tokens_file);
    fseek(loader->tokens_file, 0, SEEK_SET);
    if (loader->file_size < (B * T + 1) * sizeof(int)) {
        printf("Error: file size is too small for the batch size and sequence length\n");
        exit(EXIT_FAILURE);
    }
    loader->current_position = 0; // start at the beginning

    // allocate space for B*T + 1 integers to store the inputs and targets
    // Using CUDA CPU pinned memory for faster PCI Express transfers to GPU
    // See: https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/
    loader->batch.resize(B * T + 1);
    loader->num_batches = loader->file_size / (B * T * sizeof(int));
}

void dataloader_reset(DataLoader *loader) {
    loader->current_position = 0;
}

void dataloader_next_batch(DataLoader *loader) {
    int B = loader->B;
    int T = loader->T;
    // if we are at the end of the file, loop back to the beginning
    if (loader->current_position + (B*T+1) * sizeof(int) > loader->file_size) {
        loader->current_position = 0;
    }
    // read the B*T+1 integers from the file into batch
    fseek(loader->tokens_file, loader->current_position, SEEK_SET);
    fread(thrust::raw_pointer_cast(loader->batch.data()), sizeof(int), B*T+1, loader->tokens_file);
    // advance the current position by B*T integers
    loader->current_position += B*T * sizeof(int);
}

void dataloader_free(DataLoader *loader) {
    fclose(loader->tokens_file);
}

// ----------------------------------------------------------------------------
// sampler: takes probabilities and samples integers from them

#define GPT2_EOT 50256

unsigned int random_u32(unsigned long long *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(unsigned long long *state) { // random float32 in [0,1)
    return (random_u32(state) >> 8) / 16777216.0f;
}

int sample_mult(const thrust::host_vector<float> &probabilities, int n, float coin) {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

// ----------------------------------------------------------------------------
// Tokenizer (only supports decoding: tokens (integers) -> strings)

typedef struct {
    uint32_t vocab_size;
    char **token_table;
    int init_ok;
} Tokenizer;

void safe_printf(const char *piece) {
    // the tokens are raw bytes, and we we only want to print the printable ones
    // many bytes can be various control codes, backspace, etc.
    if (piece == NULL) { return; }
    if (piece[0] == '\0') { return; }
    // handle individual byte tokens
    // every token is asserted to be at least one byte so doing piece[1] is ok
    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) {
            return; // weird byte, don't print it
        }
    }
    printf("%s", piece);
}

void tokenizer_init(Tokenizer *tokenizer, const char *filename) {
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        // try to be more helpful as we just added this feature, erase later
        printf("---\n");
        printf("WARNING: Failed to open the tokenizer file %s\n", filename);
        printf("The Tokenizer is a new feature added April 14 2024.\n");
        printf("Re-run `python train_gpt2.py` to write it\n");
        printf("---\n");
        tokenizer->init_ok = 0;
        return;
    }
    // read in the header
    uint32_t header[256];
    freadCheck(header, sizeof(uint32_t), 256, file);
    assert(header[0] == 20240328);
    assert(header[1] == 1);
    tokenizer->vocab_size = header[2];
    // read in all the tokens
    unsigned char length;
    tokenizer->token_table = (char **)mallocCheck(tokenizer->vocab_size * sizeof(char *));
    for (uint32_t i = 0; i < tokenizer->vocab_size; i++) {
        freadCheck(&length, sizeof(unsigned char), 1, file);
        assert(length > 0); // every token should be at least one character
        char *token_bytes = (char *)mallocCheck(length + 1);
        freadCheck(token_bytes, sizeof(char), length, file);
        token_bytes[length] = '\0';  // Add null terminator for printing
        tokenizer->token_table[i] = token_bytes;
    }
    // cleanups
    fcloseCheck(file);
    tokenizer->init_ok = 1;
}

const char *tokenizer_decode(Tokenizer *tokenizer, uint32_t token_id) {
    if (tokenizer->init_ok == 0) {
        return NULL;
    }
    if (token_id < tokenizer->vocab_size) {
        return tokenizer->token_table[token_id];
    } else {
        printf("invalid token id %d!\n", token_id);
        return NULL;
    }
}

void tokenizer_free(Tokenizer *tokenizer) {
    if (tokenizer->init_ok) {
        for (uint32_t i = 0; i < tokenizer->vocab_size; i++) {
            free(tokenizer->token_table[i]);
        }
        free(tokenizer->token_table);
    }
}

// ----------------------------------------------------------------------------
// Logger lite, will probably grow/change some over time

typedef struct {
    FILE *logfile;
    int flush_every; // every how many steps to flush the log
} Logger;

void logger_init(Logger *logger, const char *filename) {
    logger->flush_every = 20;
    logger->logfile = NULL;
    if (filename != NULL) { logger->logfile = fopenCheck(filename, "w"); }
}

void logger_log_val(Logger *logger, int step, float val_loss) {
    if (logger->logfile != NULL) {
        fprintf(logger->logfile, "s:%d tel:%.4f\n", step, val_loss);
    }
}

void logger_log_train(Logger *logger, int step, float train_loss) {
    if (logger->logfile != NULL) {
        fprintf(logger->logfile, "s:%d trl:%.4f\n", step, train_loss);
        if (step % 10 == 0) { fflush(logger->logfile); }
    }
}

void logger_free(Logger *logger) {
    if (logger->logfile != NULL) { fclose(logger->logfile); }
}

// ----------------------------------------------------------------------------
// CLI, poor man's argparse

void error_usage() {
    // default run = debugging run with TinyShakespeare
    // bigger run = train on TinyStories! e.g. val/sample less often, but sample more tokens, write to logfile
    fprintf(stderr, "Usage:   ./train_gpt2cu [options]\n");
    fprintf(stderr, "Example: ./train_gpt2cu -i data/TinyStories -v 100 -s 100 -g 144 -o stories.log\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -i <string> input dataset prefix (default = data/tiny_shakespeare)\n");
    fprintf(stderr, "  -o <string> output log file (default = NULL)\n");
    fprintf(stderr, "  -b <int>    batch size B (default = 4)\n");
    fprintf(stderr, "  -t <int>    sequence length T (default = 1024)\n");
    fprintf(stderr, "  -l <float>  learning rate (default = 1e-4f)\n");
    fprintf(stderr, "  -v <int>    val_loss_every, how often we evaluate val loss (default = 20)\n");
    fprintf(stderr, "  -m <int>    val_max_batches, up to how many val batches to estimate val loss? (default = 20)\n");
    fprintf(stderr, "  -s <int>    sample_every, how often we inference the model (default = 20)\n");
    fprintf(stderr, "  -g <int>    genT, how many steps of inference we do (default = 64)\n");
    exit(EXIT_FAILURE);
}

// ----------------------------------------------------------------------------
// main training loop
int main(int argc, char *argv[]) {

    // read in the (optional) command line arguments
    const char* input_dataset_prefix = "data/tiny_shakespeare"; // or e.g. data/TinyStories
    const char* output_log_file = NULL;
    int B = 4; // batch size
    int T = 1024; // sequence length max
    float learning_rate = 1e-4f;
    int val_loss_every = 20; // every how many steps do we eval validation loss?
    int val_max_batches = 20; // how many batches max do we eval for validation loss?
    int sample_every = 20; // every how many steps to do inference?
    int genT = 64; // number of steps of inference we will do
    for (int i = 1; i < argc; i+=2) {
        if (i + 1 >= argc) { error_usage(); } // must have arg after flag
        if (argv[i][0] != '-') { error_usage(); } // must start with dash
        if (strlen(argv[i]) != 2) { error_usage(); } // must be -x (one dash, one letter)
        // read in the args
        if (argv[i][1] == 'i') { input_dataset_prefix = argv[i+1]; }
        else if (argv[i][1] == 'o') { output_log_file = argv[i+1]; }
        else if (argv[i][1] == 'b') { B = atoi(argv[i+1]); }
        else if (argv[i][1] == 't') { T = atoi(argv[i+1]); }
        else if (argv[i][1] == 'l') { learning_rate = atof(argv[i+1]); }
        else if (argv[i][1] == 'v') { val_loss_every = atoi(argv[i+1]); }
        else if (argv[i][1] == 'm') { val_max_batches = atoi(argv[i+1]); }
        else if (argv[i][1] == 's') { sample_every = atoi(argv[i+1]); }
        else if (argv[i][1] == 'g') { genT = atoi(argv[i+1]); }
        else { error_usage(); }
    }
    printf("input dataset prefix: %s\n", input_dataset_prefix);
    printf("output log file: %s\n", output_log_file == NULL ? "NULL" : output_log_file);
    printf("batch size B: %d\n", B);
    printf("sequence length T: %d\n", T);
    printf("learning rate: %f\n", learning_rate);
    printf("val_loss_every: %d\n", val_loss_every);
    printf("val_max_batches: %d\n", val_max_batches);
    printf("sample_every: %d\n", sample_every);
    printf("genT: %d\n", genT);

    // set up the device
    int deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx));
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceIdx);
    printf("[System]\n");
    printf("Device %d: %s\n", deviceIdx, deviceProp.name);

    // setup cuBLAS and cuBLASLt
    cublasCheck(cublasCreate(&cublas_handle));
    cublasCheck(cublasLtCreate(&cublaslt_handle));
    // TF32 precision is equivalent to torch.set_float32_matmul_precision('high')
    int enable_tf32 = deviceProp.major >= 8 ? 1 : 0;
    printf("enable_tf32: %d\n", enable_tf32);
    cublas_compute_type = enable_tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
    cublasMath_t cublas_math_mode = enable_tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH;
    cublasCheck(cublasSetMathMode(cublas_handle, cublas_math_mode));
    // setup the (global) cuBLASLt workspace
    thrust::device_vector<std::uint8_t> cublaslt_workspace_vec(cublaslt_workspace_size);
    cublaslt_workspace = thrust::raw_pointer_cast(cublaslt_workspace_vec.data());

    // build the GPT-2 model from a checkpoint
    GPT2 model;
    gpt2_build_from_checkpoint(model, "gpt2_124M.bin");

    // build the DataLoaders from tokens files. for now use tiny_shakespeare if available, else tiny_stories
    char train_tokens_filename[128];
    char val_tokens_filename[128];
    assert(strlen(input_dataset_prefix) < 100); // being bit lazy here, make sure we don't overflow
    sprintf(train_tokens_filename, "%s_train.bin", input_dataset_prefix);
    sprintf(val_tokens_filename, "%s_val.bin", input_dataset_prefix);

    // set up the dataloaders
    DataLoader train_loader;
    dataloader_init(&train_loader, train_tokens_filename, B, T);
    DataLoader val_loader;
    dataloader_init(&val_loader, val_tokens_filename, B, T);
    int train_num_batches = train_loader.num_batches; // let's do 1 epoch by default
    int val_num_batches = train_loader.num_batches < val_max_batches ? train_loader.num_batches : val_max_batches;
    printf("train dataset num_batches: %d\n", train_loader.num_batches);
    printf("val dataset num_batches: %d\n", val_loader.num_batches);

    // set up the logfile
    Logger logger;
    logger_init(&logger, output_log_file);

    // build the Tokenizer
    Tokenizer tokenizer;
    tokenizer_init(&tokenizer, "gpt2_tokenizer.bin");

    // some memory for generating samples from the model
    unsigned long long rng_state = 1337;

    std::unique_ptr<int[]> gen_tokens(new int[B * T]);
    thrust::host_vector<float> cpu_probs(model.config.vocab_size);

    // train
    struct timespec start, end;
    double total_sum_iteration_time_s = 0.0;
    for (int step = 0; step <= train_num_batches; step++) {
        int last_step = step == train_num_batches;

        // once in a while estimate the validation loss
        if (step % val_loss_every == 0 || last_step) {
            float val_loss = 0.0f;
            dataloader_reset(&val_loader);
            for (int i = 0; i < val_num_batches; i++) {
                dataloader_next_batch(&val_loader);
                gpt2_forward(model, val_loader.inputs(), val_loader.targets(), B, T);
                val_loss += model.mean_loss;
            }
            val_loss /= val_num_batches;
            printf("val loss %f\n", val_loss);
            logger_log_val(&logger, step, val_loss);
        }

        // once in a while do model inference to print generated text
        /*if (step > 0 && step % sample_every == 0 || last_step) {
            // fill up gen_tokens with the GPT2_EOT, which kicks off the generation
            for(int i = 0; i < B * T; ++i) {
                gen_tokens[i] = GPT2_EOT;
            }
            // now sample from the model autoregressively
            printf("generating:\n---\n");
            for (int t = 1; t < genT; t++) {
                // note that inference is very wasteful here because for each token
                // we re-calculate the forward pass for all of (B,T) positions from scratch
                // but the inference here is just for sanity checking anyway
                // and we can maybe optimize a bit more later, with careful tests
                gpt2_forward(model, gen_tokens.get(), NULL, B, T);
                // furthermore, below we're only using b=0 (i.e. the first row) of all B rows
                // we're in principle running B "inference streams" in parallel here
                // only using position 0 because it's a bit faster (copy less probs from GPU -> CPU)
                // get the V-dimensional vector probs[0, t-1, :]
                float* probs = model.acts.probs + (t-1) * model.config.vocab_size;
                // move probs back to CPU and sample
                thrust::copy_n(thrust::device_pointer_cast(probs), model.config.vocab_size, cpu_probs.begin());
                float coin = random_f32(&rng_state);
                int next_token = sample_mult(cpu_probs, model.config.vocab_size, coin);
                gen_tokens[t] = next_token;
                // print the generated token, either using the Tokenizer or a fallback
                if (tokenizer.init_ok) {
                    const char* token_str = tokenizer_decode(&tokenizer, next_token);
                    safe_printf(token_str);
                } else {
                    // fall back to printing the token id
                    printf("%d ", next_token);
                }
                fflush(stdout);
            }
            printf("\n---\n");
        }*/

        // bit confusing: we want to make sure to eval and sample on 0th iteration
        // but also after the very last iteration. so we loop for step <= train_num_batches
        // instead of just < train_num_batches (one extra due to <=), only to do
        // the validation/sampling one last time, and then we break right here as we're done.
        if (last_step) { break; }

        // do a training step
        auto start = std::chrono::steady_clock::now();
        dataloader_next_batch(&train_loader);
        gpt2_forward(model, train_loader.inputs(), train_loader.targets(), B, T);
        gpt2_zero_grad(model);
        gpt2_backward(model);
        gpt2_update(&model, learning_rate, 0.9f, 0.999f, 1e-8f, 0.0f, step+1);
        //dump_tensor( thrust::raw_pointer_cast(model.grads_memory.data()), "grads_"+std::to_string(step), model.num_parameters);
        //dump_tensor( thrust::raw_pointer_cast(model.params_memory.data()), "parameters_"+std::to_string(step), model.num_parameters);
        cudaCheck(cudaDeviceSynchronize()); // finish all CUDA work to get correct precise timings
        auto end = std::chrono::steady_clock::now();
        double time_elapsed_s = std::chrono::duration<double>(end - start).count();
        total_sum_iteration_time_s += time_elapsed_s;
        printf("step %d/%d: train loss %f (%f ms)\n", step + 1, train_num_batches, model.mean_loss, time_elapsed_s * 1000);
        logger_log_train(&logger, step, model.mean_loss);
    }
    // add a total average, for optimizations that are only mild improvements
    printf("total average iteration time: %f ms\n", total_sum_iteration_time_s / train_num_batches * 1000);

    // free
    dataloader_free(&train_loader);
    dataloader_free(&val_loader);
    tokenizer_free(&tokenizer);
    cublasCheck(cublasDestroy(cublas_handle));
    cublasCheck(cublasLtDestroy(cublaslt_handle));
    logger_free(&logger);

    return 0;
}
#endif
