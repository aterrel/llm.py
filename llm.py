import math
import os
import struct
import sys
import time

import cupy
import cupyx
import numba.cuda
import numpy

import cuda.cooperative.experimental as cudax
from pynvjitlink import patch
from cuda import cuda, cudart, nvrtc
import nvmath.linalg
from nvmath.bindings import cublas
from nvmath.bindings import cublasLt as cublaslt


cublaslt_workspace_size = 32 * 1024 * 1024
cublaslt_workspace = None
cublas_compute_type = None
cublas_handle = None
cublaslt_handle = None

def print4(data):
    print(data[0], data[256], data[512])


def _cudaGetErrorEnum(error):
    if isinstance(error, cuda.CUresult):
        err, name = cuda.cuGetErrorName(error)
        return name if err == cuda.CUresult.CUDA_SUCCESS else "<unknown>"
    elif isinstance(error, cudart.cudaError_t):
        return cudart.cudaGetErrorName(error)[1]
    elif isinstance(error, nvrtc.nvrtcResult):
        return nvrtc.nvrtcGetErrorString(error)[1]
    else:
        raise RuntimeError('Unknown error type: {}'.format(error))


def checkCudaErrors(result):
    if result[0].value:
        raise RuntimeError("CUDA error code={}({})".format(result[0].value, _cudaGetErrorEnum(result[0])))
    if len(result) == 1:
        return None
    elif len(result) == 2:
        return result[1]
    else:
        return result[1:]

"""
--------------------- FORWARD KERNELS ---------------------
"""

def compare_tensor(cupy_val, name, size):
    if False:
        numpy_array = cupy.asnumpy(cupy_val[:size])
        file_data = numpy.fromfile(f"outs/{name}", dtype=numpy.float32)
        print("Checking ", name)
        numpy.testing.assert_allclose(file_data, numpy_array, rtol=1e-04, atol=1e-05)

patch.patch_numba_linker(lto=True)        
cx_warp_sum = cudax.warp.sum(numba.float32)
sum_storage_bytes = cx_warp_sum.temp_storage_bytes
cx_warp_files = cx_warp_sum.files

def max_op(a, b):
    return a if a > b else b

cx_warp_max = cudax.warp.reduce(numba.float32, max_op)
max_storage_bytes = cx_warp_max.temp_storage_bytes
cx_warp_files += cx_warp_max.files

# TODO use the cuda.cooperative.experimental.warp.sum API instead
@numba.cuda.jit(device=True, fastmath=True)
def warp_sum(meta_group_rank, thread_rank, warp_size, sum):
    temp_storage = numba.cuda.shared.array(shape=sum_storage_bytes, dtype=numba.uint8)
    warp_output = cx_warp_sum(temp_storage, sum)
    return numba.cuda.shfl_sync(0xFFFFFFFF, warp_output, 0)


@numba.cuda.jit(device=True, fastmath=True)
def warp_max(meta_group_rank, thread_rank, warp_size, val):
    temp_storage = numba.cuda.shared.array(shape=max_storage_bytes, dtype=numba.uint8)
    warp_output = cx_warp_max(temp_storage, val)
    return numba.cuda.shfl_sync(0xFFFFFFFF, warp_output, 0)


@numba.cuda.jit(fastmath=True, link=cx_warp_files)
def layernorm_forward_kernel3(out, mean, rstd, inp, weight, bias, N, C):
    # This emulates the warp cg which is not available in numba
    warp_size = 32
    meta_group_size = numba.cuda.blockDim.x // warp_size;
    meta_group_rank = numba.cuda.threadIdx.x // warp_size;
    thread_rank = numba.cuda.threadIdx.x % warp_size
    idx = numba.cuda.blockIdx.x * meta_group_size + meta_group_rank
    # Ensure the thread is within the bounds of the array
    if idx >= N:
        return
    # Space so all the threads can share partial data
    sums = numba.cuda.shared.array(shape=512, dtype=numba.float32)

    #mean
    sum = 0
    x = inp[idx * C:]
    for i in range(thread_rank, C, warp_size):
        sum += x[i]

    m = warp_sum(meta_group_rank, thread_rank, warp_size, sum) / C
    if (thread_rank == 0):
        mean[idx] = m

    # rstd
    sum = 0
    for i in range(thread_rank, C, warp_size):
        diff = x[i] - m
        sum += diff * diff

    s = warp_sum(meta_group_rank, thread_rank, warp_size, sum)
    s = numba.cuda.libdevice.rsqrt(s / C + 1e-5)

    if (thread_rank == 0):
        rstd[idx] = s

    # final normalization and scaling by weight/bias
    o = out[idx * C:]
    for c in range(thread_rank, C, warp_size):
        n = s  * (x[c] - m)
        o[c] = n * weight[c] + bias[c]

def ceil_div(a, b):
    return -(a // -b)

def layernorm_forward(out, mean,  rstd, inp, weight, bias, B, T, C):
    block_size = 512
    N = B * T
    grid_size = ceil_div(N * 32, block_size)
    layernorm_forward_kernel3[grid_size, block_size](out, mean, rstd, inp, weight, bias, N, C)


def matmul_forward_cublas(out, inp, weight, bias, B, T, C, OC):
    assert(bias == None)  # bias is not supported for this kernel
    alpha = numpy.array(1.0, dtype=numpy.float32)
    beta = numpy.array(0.0, dtype=numpy.float32)
    cublas.sgemm(cublas_handle, cublas.Operation.T, cublas.Operation.N, OC, B*T, C, alpha.ctypes.data, weight.data.ptr, C, inp.data.ptr, C, beta.ctypes.data, out.data.ptr, OC)


def cublaslt_setattr(matmul_desc, name, value):
    name = name.upper()
    DescEnum = cublaslt.MatmulDescAttribute 
    scalar_attrs =  [e.name for e in DescEnum] 
    if name not in scalar_attrs:
        raise RuntimeError("Unknown attr")
    get_dtype = cublaslt.get_matmul_desc_attribute_dtype
    attribute_buffer = numpy.zeros((1,), dtype=get_dtype(DescEnum[name]))
    attribute_buffer[0] = value
    cublaslt.matmul_desc_set_attribute(matmul_desc, DescEnum[name].value, attribute_buffer.ctypes.data, attribute_buffer.itemsize)

def cublaslt_set_preference_attr(preference, name, value):
    name = name.upper()
    PreferenceEnum = cublaslt.MatmulPreferenceAttribute
    scalar_attrs =  [e.name for e in PreferenceEnum] 
    if name not in scalar_attrs:
        raise RuntimeError("Unknown attr")
    get_dtype = cublaslt.get_matmul_preference_attribute_dtype
    attribute_buffer = numpy.zeros((1,), dtype=get_dtype(PreferenceEnum[name]))
    attribute_buffer[0] = value
    cublaslt.matmul_preference_set_attribute(preference, PreferenceEnum[name].value, attribute_buffer.ctypes.data, attribute_buffer.itemsize)


def matmul_forward_cublaslt(out, inp, weight, bias, B, T, C, OC):
    has_bias = (bias is not None);

    # check bias alignment
    if bias.data.ptr % 16 != 0:
        raise RuntimeError("Bias pointer is not aligned (cuBLASLt requirement)!\n")
    #options = nvmath.linalg.advanced.MatmulOptions(compute_type=nvmath.linalg.advanced.MatmulComputeType.COMPUTE_32F)
    #weight = weight[:C*OC].reshape((OC, C))
    #inp = inp[:B*T*C].reshape((C,B*T))
    #print(weight.shape, inp.shape)
    #if has_bias:
    #    bias = bias[:OC].reshape((OC,))
    #nvmath.linalg.advanced.Matmul(weight, inp, c=bias, alpha=1.0, beta=0.0, options=options)
    returnedResults = 0
    # create the operation descriptor
    opNoTranspose = cublas.Operation.N
    opTranspose = cublas.Operation.T
    epilogueBias = cublaslt.Epilogue.BIAS
    cuda_r_32f = nvmath.CudaDataType.CUDA_R_32F
    operation_desc = cublaslt.matmul_desc_create(cublas_compute_type, cuda_r_32f)
    cublaslt_setattr(operation_desc, "TRANSA", opTranspose)
    cublaslt_setattr(operation_desc, "TRANSB", opNoTranspose)
    cublaslt_setattr(operation_desc, "EPILOGUE", epilogueBias)
    if has_bias:
        cublaslt_setattr(operation_desc, "BIAS_POINTER", bias.data.ptr)
    else:
        cublaslt_setattr(operation_desc, "BIAS_POINTER", 0)
    weight_layout = cublaslt.matrix_layout_create(cuda_r_32f, C, OC, C) 
    input_layout = cublaslt.matrix_layout_create(cuda_r_32f, C, B*T, C) 
    output_layout = cublaslt.matrix_layout_create(cuda_r_32f, OC, B*T, OC) 
    bias_layout = cublaslt.matrix_layout_create(cuda_r_32f, OC, 1, OC) 
    preference = cublaslt.matmul_preference_create()
    cublaslt_set_preference_attr(preference, "MAX_WORKSPACE_BYTES", cublaslt_workspace_size)


    # find a suitable algorithm
    algorithm_dtype = algorithm_dtype = numpy.dtype([('algorithm', numpy.uint64, (8,)), ('workspace_size', numpy.uint64), ('status', numpy.int32), ('waves_count', numpy.float32), ('reserved', numpy.int32, (4,))])
    algorithms_buffer  = numpy.zeros((1,), dtype=algorithm_dtype)
    num_algorithms = numpy.zeros((1,), dtype=numpy.int32)
    cublaslt.matmul_algo_get_heuristic(
        cublaslt_handle, operation_desc, weight_layout, input_layout, output_layout, output_layout, preference, 1, algorithms_buffer.ctypes.data, num_algorithms.ctypes.data
    )
    if num_algorithms[0] == 0:
        raise RuntimeError(f"No cuBLASLt algorithm: B: {B}, T: {T}, C: {C}, OC: {OC}, bias: {has_bias}")

    # call matmul
    alpha = numpy.array(1.0, dtype=numpy.float32)
    beta = numpy.array(0.0, dtype=numpy.float32)
    cublaslt.matmul(cublaslt_handle, operation_desc,
        alpha.ctypes.data, weight.data.ptr, weight_layout, inp.data.ptr, input_layout, beta.ctypes.data,
        out.data.ptr, output_layout, out.data.ptr, output_layout, algorithms_buffer[0]['algorithm'].ctypes.data,
        cublaslt_workspace.data.ptr, cublaslt_workspace_size, 0)

    cublaslt.matmul_preference_destroy(preference)
    cublaslt.matmul_desc_destroy(operation_desc)
    cublaslt.matrix_layout_destroy(weight_layout)
    cublaslt.matrix_layout_destroy(input_layout)
    cublaslt.matrix_layout_destroy(output_layout)
    cublaslt.matrix_layout_destroy(bias_layout)



@numba.cuda.jit(fastmath=True, link=cx_warp_files)
def softmax_forward_kernel5(out, inv_temperature, inp, N, T):
    # inp, out shape: (N, T, T), where N = B * NH
    # fuses the multiplication by scale inside attention
    # directly autoregressive, so we only compute the lower triangular part
    # uses the online softmax algorithm
    assert(T % 4 == 0) 

    # This emulates the warp cg which is not available in numba
    warp_size = 32
    meta_group_size = numba.cuda.blockDim.x // warp_size;
    meta_group_rank = numba.cuda.threadIdx.x // warp_size;
    thread_rank = numba.cuda.threadIdx.x % warp_size
    idx = (numba.cuda.gridDim.x - numba.cuda.blockIdx.x -1) * meta_group_size + meta_group_rank  # backward order
    if idx >= N * T:
        return
    own_pos = idx % T
    pos_by_4 = own_pos // 4
    x = inp[idx * T:]
    maxval = -3.402823e+38  # FLT_MAX
    sumval = 0.0
    for i in range(thread_rank , pos_by_4, warp_size):
        v = x[i*4:i*4+4]
        old_maxval = maxval
        for k in range(4):
            maxval = numba.cuda.libdevice.fmaxf(maxval, v[k])
        sumval *= numba.cuda.libdevice.expf(inv_temperature * (old_maxval - maxval))
        for k in range(4):
            sumval += numba.cuda.libdevice.expf(inv_temperature * (v[k] - maxval))
    if (4 * pos_by_4 + thread_rank) <= own_pos:
        old_maxval = maxval
        maxval = numba.cuda.libdevice.fmaxf(maxval, x[4*pos_by_4 + thread_rank])
        sumval *= numba.cuda.libdevice.expf(inv_temperature * (old_maxval - maxval))
        sumval += numba.cuda.libdevice.expf(inv_temperature * (x[4*pos_by_4 + thread_rank] - maxval))

    global_maxval = warp_max(meta_group_rank, thread_rank, warp_size, maxval)
    sumval *= numba.cuda.libdevice.expf(inv_temperature * (maxval - global_maxval))

    # reduce sumval
    sum = warp_sum(meta_group_rank, thread_rank, warp_size, sumval)
    # divide the whole row by the sum
    norm = 1.0 / sum

    for i in range(thread_rank, own_pos + 1, warp_size):
        # recalculation is faster than doing the round-trip through memory
        ev = numba.cuda.libdevice.expf(inv_temperature * (x[i] - global_maxval))
        out[idx * T + i] = ev * norm


@numba.cuda.jit(device=True, fastmath=True)
def i2n_4(idx, E1, E2, E3):
    b = idx // (E1 * E2 * E3)
    rest = idx % (E1 * E2 * E3)
    nh_ = rest // (E2 * E3)
    rest = rest % (E2 * E3)
    t = rest // E3
    hs = rest % E3
    return (b, t, nh_, hs)


@numba.cuda.jit(fastmath=True)
def qkv_inp_kernel(q, k, v, inp, NH, T, HS, size):
    idx = numba.cuda.grid(1)
    # Ensure the thread is within the bounds of the array
    if idx < size:
        b, t, nh_, hs = i2n_4(idx, NH, T, HS)
        q[idx] = inp[b, t, 0, nh_, hs]
        k[idx] = inp[b, t, 1, nh_, hs]
        v[idx] = inp[b, t, 2, nh_, hs]

   
def qkv_inp(q, k, v, inp, NH, T, HS, size):
    threads_per_block = 256  
    blocks_per_grid = (size + threads_per_block - 1) // threads_per_block
    qkv_inp_kernel[blocks_per_grid, threads_per_block](q, k, v, inp, NH, T, HS, size)


@numba.cuda.jit(fastmath=True)
def scatter_kernel(out, vaccum, NH, T, HS, size):
    idx = numba.cuda.grid(1)
    # Ensure the thread is within the bounds of the array
    if idx < size:
        b, n, nh_, d_ = i2n_4(idx, NH, T, HS)
        out[(b * NH * T * HS) + (n * NH * HS) + (nh_ * HS) + d_] = vaccum[idx]
    
def scatter(out, vaccum, NH, T, HS, size):
    threads_per_block = 256  
    blocks_per_grid = (size + threads_per_block - 1) // threads_per_block
    scatter_kernel[blocks_per_grid, threads_per_block](out, vaccum, NH, T, HS, size)
  

def attention_forward(out, vaccum, qkvr, preatt, att, inp, B, T, C, NH):
    block_size = 256
    softmax_block_size = 256
    HS = C // NH  # head size

    q = qkvr[0 * B * T * C:]
    k = qkvr[1 * B * T * C:]
    v = qkvr[2 * B * T * C:]

    inp_md = inp[:B*T*3*NH*HS].reshape((B, T, 3, NH, HS))
    size = B * NH * T * HS
    # Q[b][nh_][n][d_] = inp[b][n][0][nh_][d_]  
    qkv_inp(q, k, v, inp_md, NH, T, HS, size)

    # batched matrix multiply using cuBLAS
    alpha = numpy.array(1.0, dtype=numpy.float32)
    beta = numpy.array(0.0, dtype=numpy.float32)
    cublas.sgemm_strided_batched(cublas_handle, cublas.Operation.T, cublas.Operation.N, T, T, HS, alpha.ctypes.data, k.data.ptr, HS, T * HS, q.data.ptr, HS, T*HS, beta.ctypes.data, preatt.data.ptr, T, T*T, B*NH)

    scale = 1.0 / math.sqrt(HS);
    grid_size = ceil_div(B * NH * T * 32, softmax_block_size);
    softmax_forward_kernel5[grid_size, softmax_block_size](att, scale, preatt, B * NH, T)
    # new approach: first cuBLAS another batched matmul
    # y = att @ v # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
    cublas.sgemm_strided_batched(cublas_handle, cublas.Operation.N, cublas.Operation.N, HS, T, T, alpha.ctypes.data, v.data.ptr, HS, T * HS, att.data.ptr, T, T*T, beta.ctypes.data, vaccum.data.ptr, HS, T*HS, B*NH)
      
    # now unpermute
    # y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
    scatter(out, vaccum, NH, T, HS, B * T * C)


@numba.cuda.jit(device=True, fastmath=True)
def i2n_2(idx, E1, E2):
    bt = idx // E1
    b = bt // E2
    t = bt % E2
    c = idx % E1
    return (b, t, c)


def residual_forward(out, inp1, inp2, N):
    cupy.add(inp1[:N], inp2[:N], out=out[:N])


@numba.cuda.jit(fastmath=True)
def encoder_forward_kernel(out_md, wte_md, wpe_md, inp_md, C, T):
    idx = numba.cuda.grid(1)
    # Ensure the thread is within the bounds of the array
    if idx < out_md.size:
        b, t, c = i2n_2(idx, C, T)
        out_md[b, t, c] = wte_md[inp_md[b, t], c] + wpe_md[t, c]
   

def encoder_forward(out, inpv, wte, wpe, B, T, C, V):
    out_md = out.reshape(B, T, C)
    wte_md = wte.reshape(V, C)
    wpe_md = wpe.reshape(T, C)
    inp_md = inpv.reshape(B, T)
    threads_per_block = 256  
    blocks_per_grid = (out.size + threads_per_block - 1) // threads_per_block
    encoder_forward_kernel[blocks_per_grid, threads_per_block](out_md, wte_md, wpe_md, inp_md, C, T)

@numba.cuda.jit(fastmath=True)
def gelu_forward_kernel(out, inp, N):
    idx = numba.cuda.grid(1)
    if idx >= N:
        return
    xi = inp[idx]
    cube = 0.044715 * xi * xi * xi
    scaling_factor = numba.cuda.libdevice.sqrtf(2.0 / math.pi)
    out[idx] = 0.5 * xi * (1.0 + numba.cuda.libdevice.tanhf(scaling_factor * (xi + cube)))


def gelu_forward(out, inp, N):
    threads_per_block = 256  
    blocks_per_grid = (out.size + threads_per_block - 1) // threads_per_block
    gelu_forward_kernel[blocks_per_grid, threads_per_block](out, inp, N)

@numba.cuda.jit(device=True, fastmath=True, link=cx_warp_files)
def prepare_softmax_blockwide_nofloat4(idx, inp, V, P):
    x = inp[idx * P:]
    thread_maxval = -math.inf
    thread_sumval = 0.0

    # do the loop in reverse to maximise probability of L2 cache hits
    # so even small L2s get some hits on the 2nd read of the same thread
    for i in range(V + numba.cuda.threadIdx.x - numba.cuda.blockDim.x, -1, -numba.cuda.blockDim.x):
        v = x[i]
        old_maxval = thread_maxval
        thread_maxval = numba.cuda.libdevice.fmaxf(thread_maxval, v)
        thread_sumval *= numba.cuda.libdevice.expf(old_maxval - thread_maxval)
        thread_sumval += numba.cuda.libdevice.expf(v - thread_maxval)

    # two reductions of up to 1024 threads:
    # 1) inside warp (shuffle), 2) cross-warp (shared memory), 3) inside warp (shuffle)
    # this results in much cleaner assembly than a multi-warp cg::reduce
    shared_maxval = numba.cuda.shared.array(shape=32, dtype=numba.float32)
    shared_sumval = numba.cuda.shared.array(shape=32, dtype=numba.float32)
    num_warps = numba.cuda.blockDim.x // 32
    warp_id = numba.cuda.threadIdx.x // 32
    lane_id = numba.cuda.threadIdx.x % 32

    # reduce maxval within each warp
    warp_maxval = warp_max(warp_id, lane_id, 32, thread_maxval)
    # thread 0 in each warp writes to shared memory
    if lane_id == 0:
        shared_maxval[warp_id] = warp_maxval
    numba.cuda.syncthreads()

    # each thread now loads the maxval across previous warps
    # if the thread is "out of range" of data, use -FLT_MAX as the maxval
    warp_maxval = shared_maxval[lane_id] if (lane_id < num_warps) else -3.402823e+38  # FLT_MAX     
    block_maxval = warp_max(warp_id, lane_id, 32, warp_maxval)
    # each thread uses maxval to scale sumval to avoid numerical instability / overflow
    thread_sumval *= numba.cuda.libdevice.expf(thread_maxval - block_maxval)
    # (warp-level) reduce sumval, thread 0 in each warp saves result in shared memory
    warp_sumval = warp_sum(warp_id, lane_id, 32, thread_sumval)
    if lane_id == 0:
        shared_sumval[warp_id] = warp_sumval
    numba.cuda.syncthreads()
    # same strategy, now reduce sumval across warps
    warp_sumval = shared_sumval[lane_id] if (lane_id < num_warps) else 0.0
    block_sumval = warp_sum(warp_id, lane_id, 32, warp_sumval)
    return (1.0 / block_sumval, block_maxval)


@numba.cuda.jit(fastmath=True, link=cx_warp_files)
def fused_classifier_kernel3(logits, losses, targets, B, T, V, P):
    idx = numba.cuda.blockIdx.x
    ix = targets[idx]
    # softmax (reading B * T * V, same logits read again below, hopefully still in cache)
    scale, offset = prepare_softmax_blockwide_nofloat4(idx, logits, V, P)
    # calculate the probability needed for the loss and update (single-threaded)
    if numba.cuda.threadIdx.x == 0:
        prob = numba.cuda.libdevice.expf(logits[idx * P + ix] - offset) * scale;
        losses[idx] = -numba.cuda.libdevice.logf(prob);
    # very sensible default for dlosses is 1/(B*T), which is the uniform loss
    dloss = 1.0 / (B * T)
    # calculate the gradients directly, saves bandwidth from probs during training
    # but also supports writing probs for inference-only and debugging
    logits_vec = logits[idx * P:]
    for i in range(numba.cuda.threadIdx.x, V, numba.cuda.blockDim.x):
        # this is the 2nd read of logits after the one in prepare_softmax2
        # this data will never be needed again, so we reduce cache persistence
        v = logits_vec[i]
        prob = numba.cuda.libdevice.expf(v - offset) * scale
        indicator = 1.0 if i == ix else 0.0
        logits[idx * P + i] = (prob - indicator) * dloss


# replaces logits with logit gradients
def fused_classifier3(logits, losses, targets, B, T, V, P):
    block_size = 1024;
    N = B * T;
    grid_size = N;
    fused_classifier_kernel3[grid_size, block_size](logits, losses, targets, B, T, V, P)

"""
------------------------------------------------------------
--------------------- BACKWARD KERNELS ---------------------
"""

@numba.cuda.jit(fastmath=True, link=cx_warp_files)
def matmul_backward_bias_kernel2(dbias, dout, B, T, OC):
    # dout is (B, T, OC), dbias is (OC)
    # e.g. if block_size = 128, then we have 4 warps per block, each in charge of one output channel
    warp_size = 32
    meta_group_size = numba.cuda.blockDim.x // warp_size;
    meta_group_rank = numba.cuda.threadIdx.x // warp_size;
    thread_rank = numba.cuda.threadIdx.x % warp_size
    # meta_group_size is the number of warps in a block (e.g. 4), meta_group_rank is the warp index (0,1,2,3)
    idx = numba.cuda.blockIdx.x * meta_group_size + meta_group_rank
    if idx >= OC:
        return

    BT = B * T  # number of elements to reduce in total, per channel
    sum = 0.0
    # first, thread coarsening to sum reduce the problem size from B*T to 32
    for i in range(thread_rank , BT, warp_size):
        sum += dout[i * OC + idx]
    # now do a warp-level reduce to get the sum across the 32 threads in this warp
    sum = warp_sum(meta_group_rank, thread_rank, warp_size, sum)
    # write the result to output (global memory)
    if thread_rank == 0:
        dbias[idx] += sum


def matmul_backward(dinp, dweight, dbias, dout, inp, weight, B, T, C, OC):
    one = numpy.array(1.0, dtype=numpy.float32)
    zero = numpy.array(0.0, dtype=numpy.float32)
    # backward to input, uses = in the backward pass (set the gradient)
    cublas.sgemm(cublas_handle, cublas.Operation.N, cublas.Operation.N, C, B*T, OC, one.ctypes.data, weight.data.ptr, C, dout.data.ptr, OC, zero.ctypes.data, dinp.data.ptr, C)
    # backward to weight, uses += in the backward pass (accumulate the gradient)
    cublas.sgemm(cublas_handle, cublas.Operation.N, cublas.Operation.T, C, OC, B*T, one.ctypes.data, inp.data.ptr, C, dout.data.ptr, OC, one.ctypes.data, dweight.data.ptr, C)
    # backward to bias, if given, does a +=
    if dbias is not None:
        block_size = 512
        grid_size = ceil_div(OC * 32, block_size)
        matmul_backward_bias_kernel2[grid_size, block_size](dbias, dout, B, T, OC)


@numba.cuda.jit(fastmath=True, link=cx_warp_files)
def layernorm_backward_kernel(dinp, dweight, dbias, dout, inp, weight, mean, rstd, B, T, C):
    warp_size = 32
    meta_group_size = numba.cuda.blockDim.x // warp_size;
    meta_group_rank = numba.cuda.threadIdx.x // warp_size;
    thread_rank = numba.cuda.threadIdx.x % warp_size
    idx = numba.cuda.blockIdx.x * meta_group_size + meta_group_rank
    N = B * T
    if idx >= N:
        return 
    b = idx // T
    t = idx % T
    dout_bt = dout[b * T * C + t * C:] 
    inp_bt = inp[b * T * C + t * C:] 
    dinp_bt = dinp[b * T * C + t * C:] 
    mean_bt = mean[b * T + t]
    rstd_bt = rstd[b * T + t]

    # first: two reduce operations
    dnorm_mean = 0.0
    dnorm_norm_mean = 0.0
    for i in range(thread_rank, C, warp_size):
        norm_bti = (inp_bt[i] - mean_bt) * rstd_bt
        dnorm_i = weight[i] * dout_bt[i] 
        dnorm_mean += dnorm_i
        dnorm_norm_mean =+ dnorm_i * norm_bti

    dnorm_mean = warp_sum(meta_group_rank, thread_rank, warp_size, dnorm_mean)
    dnorm_norm_mean = warp_sum(meta_group_rank, thread_rank, warp_size, dnorm_norm_mean)

    dnorm_mean = dnorm_mean / C
    dnorm_norm_mean = dnorm_norm_mean / C
    for i in range(thread_rank, C, warp_size):
        norm_bti = (inp_bt[i] - mean_bt) * rstd_bt 
        dnorm_i = weight[i] * dout_bt[i]
        # gradient contribution to bias
        numba.cuda.atomic.add(dbias, i, dout_bt[i])
        # gradient contribution to weight
        numba.cuda.atomic.add(dweight, i, norm_bti * dout_bt[i])
        # gradient contribution to input
        dval = 0.0
        dval += dnorm_i  # term 1 
        dval -= dnorm_mean  # term 2
        dval -= norm_bti * dnorm_norm_mean  # term 3
        dval *= rstd_bt  # final scale
        dinp_bt[i] += dval 


def layernorm_backward(dinp, dweight, dbias, dout, inp, weight, mean, rstd, B, T, C):
    block_size = 256
    N = B * T
    # one warp per token, so we need to divide by 32 here.
    grid_size = ceil_div(N, block_size // 32)
    layernorm_backward_kernel[grid_size, block_size](dinp, dweight, dbias, dout, inp, weight, mean, rstd, B, T, C)


@numba.cuda.jit(fastmath=True)
def gelu_backward_kernel(dinp, inp, dout, N):
    i = numba.cuda.grid(1)
    if (i < N):
        scaling_factor = numba.cuda.libdevice.sqrtf(2.0 / math.pi)
        x = inp[i]
        cube =  0.044715 * x * x * x
        tanh_arg = scaling_factor * (x + cube)
        tanh_out = numba.cuda.libdevice.tanhf(tanh_arg)
        coshf_out = numba.cuda.libdevice.coshf(tanh_arg)
        sech_out = 1.0 / (coshf_out * coshf_out)
        local_grad = 0.5 * (1.0 + tanh_out) + x * 0.5 * sech_out *scaling_factor * (1.0 + 3.0 * 0.044715 *x *x)
        dinp[i] = local_grad * dout[i]


def gelu_backward(dinp, inp, dout, N):
    block_size = 128;
    grid_size = ceil_div(N, block_size)
    gelu_backward_kernel[grid_size, block_size](dinp, inp, dout, N)


@numba.cuda.jit(fastmath=True)
def unpermute_kernel_backward(dinp, dout, B, N, NH, d):
    idx = numba.cuda.grid(1)
    if idx < (B * NH * N * d):
        b = idx // (NH * N * d)
        rest = idx % (NH * N * d)
        nh_ = rest // (N * d)
        rest = rest % (N * d)
        n = rest // d
        d_ = rest % d
        other_idx = (b * NH * N * d) + (n * NH * d) + (nh_ * d) + d_
        dinp[idx] = dout[other_idx]



@numba.cuda.jit(fastmath=True, link=cx_warp_files)
def softmax_autoregressive_backward_kernel(dpreatt, datt, att, B, T, C, scale):
    BlockSize = 256
    T_per_block = 4

    warp_size = 32
    meta_group_size = numba.cuda.blockDim.x // warp_size;
    meta_group_rank = numba.cuda.threadIdx.x // warp_size;
    thread_rank = numba.cuda.threadIdx.x % warp_size
    block_acc = numba.cuda.shared.array(shape=32, dtype=numba.float32)

    idx = numba.cuda.blockIdx.y 
    # go through blocks in reverse order, so the slowest block starts first
    t0 = T - 1 - T_per_block * numba.cuda.blockIdx.x
    att = att[idx * T * T:]
    datt = datt[idx * T * T:]
    dpreatt = dpreatt[idx * T * T:]

    if meta_group_rank == 0:
       block_acc[thread_rank] = 0

    for to in range(0, T_per_block):
       t = t0 - to
       if t < 0:
           return
       att_bth = att[t*T:] 
       datt_bth = datt[t*T:] 
       dpreatt_bth = dpreatt[t*T:] 

       local_sum = 0
       for t2 in range(numba.cuda.threadIdx.x, t + 1, BlockSize):
           local_sum += att_bth[t2] * datt_bth[t2]
           #if numba.cuda.blockIdx.x == 0 and numba.cuda.blockIdx.y == 0 and numba.cuda.threadIdx.x == 0:
           #    print(t2 + t *T + idx * T * T, local_sum, att_bth[t2], datt_bth[t2])

       block_acc[meta_group_rank] = warp_sum(meta_group_rank, thread_rank, warp_size, local_sum) 
       numba.cuda.syncthreads()
       local_sum = warp_sum(meta_group_rank, thread_rank, warp_size, block_acc[thread_rank]) 
       for t3 in range(numba.cuda.threadIdx.x, t + 1, BlockSize):
            acc = att_bth[t3] * (datt_bth[t3] - local_sum)
            dpreatt_bth[t3] = scale * acc


@numba.cuda.jit(fastmath=True)
def permute_kernel_backward(dinp, dq, dk, dv, B, N, NH, d):
    idx = numba.cuda.grid(1)
    if (idx < B * NH * N * d):
        b = idx // (NH * N * d)
        rest = idx % (NH * N * d)
        nh_ = rest // (N * d)
        rest = rest % (N * d)
        n = rest // d
        d_ = rest % d

        inp_idx = (b * N * 3 * NH * d) + (n * 3 * NH * d) + (0 * NH * d) + (nh_ * d) + d_
        dinp[inp_idx] = dq[idx]
        dinp[inp_idx + NH * d] = dk[idx]
        dinp[inp_idx + 2 * (NH * d)] = dv[idx]


# the sequence of transformations in this compound op is:
# inp (B,T,3C) -> qkvr (B,T,3C) -> preatt (B,NH,T,T) -> att (B,NH,T,T) -> vaccum (B,T,C) -> out (B,T,C)
def attention_backward(dinp, dqkvr, dpreatt, datt, dvaccum, dout, inp, qkvr, att, B, T, C, NH):
    block_size = 256
    HS = C // NH  # head size
    one = numpy.array(1.0, dtype=numpy.float32)
    zero = numpy.array(0.0, dtype=numpy.float32)
    # unpack convenience pointers into q, k, v
    q = qkvr[0 * B * T * C:] 
    k = qkvr[1 * B * T * C:] 
    v = qkvr[2 * B * T * C:] 
    dq = dqkvr[0 * B * T * C:] 
    dk = dqkvr[1 * B * T * C:] 
    dv = dqkvr[2 * B * T * C:] 
    # backward through the unpermute operation
    num_blocks = ceil_div(B * T * C, block_size)
    unpermute_kernel_backward[num_blocks, block_size](dvaccum, dout, B, T, NH, HS)
    # backward into datt
    cublas.sgemm_strided_batched(cublas_handle, cublas.Operation.T, cublas.Operation.N, T, T, HS, one.ctypes.data, v.data.ptr, HS, T * HS, dvaccum.data.ptr, HS, T * HS, zero.ctypes.data, datt.data.ptr, T, T * T, B * NH)
    # backward into dv
    cublas.sgemm_strided_batched(cublas_handle, cublas.Operation.N, cublas.Operation.T, HS, T, T, one.ctypes.data, dvaccum.data.ptr, HS, T * HS, att.data.ptr, T, T * T, zero.ctypes.data, dv.data.ptr, HS, T * HS, B * NH)
    # backward into preatt
    scale = 1.0 / math.sqrt(HS);
    #print4(dpreatt)
    softmax_autoregressive_backward_kernel[(T // 4, B * NH), 256](dpreatt, datt, att, B, T, C, scale)
    #print4(dpreatt)
    #print4(datt)
    #print4(att)
    #sys.exit(0)
    # backward into q
    cublas.sgemm_strided_batched(cublas_handle, cublas.Operation.N, cublas.Operation.N, HS, T, T, one.ctypes.data, k.data.ptr, HS, T * HS, dpreatt.data.ptr, T, T * T, zero.ctypes.data, dq.data.ptr, HS, T * HS, B * NH)
    # backward into k
    cublas.sgemm_strided_batched(cublas_handle, cublas.Operation.N, cublas.Operation.T, HS, T, T, one.ctypes.data, q.data.ptr, HS, T * HS, dpreatt.data.ptr, T, T * T, zero.ctypes.data, dk.data.ptr, HS, T * HS, B * NH)
    # backward into inp
    num_blocks = ceil_div(B * NH * T * HS, block_size);
    permute_kernel_backward[num_blocks, block_size](dinp, dq, dk, dv, B, T, NH, HS)


@numba.cuda.jit(fastmath=True)
def encoder_backward_kernel(dwte, dwpe, dout, inp, B, T, C):
    idx = numba.cuda.grid(1)
    N = B * T * C;
    if (idx < N):
        bt = idx // C;
        b = bt // T;
        t = bt % T;
        c = idx % C;

        ix = inp[b * T + t]

        dout_btc = dout[b * T * C + t * C + c:]
        dwte_ix = dwte[ix * C + c:]
        dwpe_tc = dwpe[t * C + c:]
        numba.cuda.atomic.add(dwte_ix, 0, dout_btc[0])
        numba.cuda.atomic.add(dwpe_tc, 0, dout_btc[0])


def encoder_backward(dwte, dwpe, dout, inp, B, T, C):
    N = B * T * C;
    block_size = 256;
    grid_size = ceil_div(N, block_size);
    encoder_backward_kernel[grid_size, block_size](dwte, dwpe, dout, inp, B, T, C)

"""
------------------------------------------------------------
"""
# Implements linear interpolation using only two floating-point operations (as opposed to three in a naive implementation).
# Reference: https://developer.nvidia.com/blog/lerp-faster-cuda
@numba.cuda.jit(device=True, fastmath=True)
def lerp_(start, end, weight):
    return numba.cuda.libdevice.fma(weight, end, numba.cuda.libdevice.fma(-weight, start, start))


@numba.cuda.jit(fastmath=True)
def adamw_kernel2(params_memory, grads_memory, m_memory, v_memory, num_parameters,
                  learning_rate, beta1, beta2, beta1_correction, beta2_correction, eps, weight_decay):
    i = numba.cuda.grid(1)
    if i >= num_parameters:
        return;  # guard
    grad = grads_memory[i]
    m = m_memory[i]
    v = v_memory[i]
    # update the first moment (momentum)
    m = lerp_(grad, m, beta1)
    m_memory[i] = m
    # update the second moment (RMSprop)
    v = lerp_(grad * grad, v, beta2)
    v_memory[i] = v
    m /= beta1_correction  # m_hat
    v /= beta2_correction  # v_hat
    params_memory[i] -= learning_rate * (m / (numba.cuda.libdevice.sqrtf(v) + eps) + weight_decay * params_memory[i])

"""
------------------------------------------------------------
"""

#TODO(ecastill) Use torch DataLoader instead?
class DataLoader:
    def __init__(self):
        self.B = 0
        self.T = 0
        self.tokens_file = None
        self.file_size = 0
        self.current_position = 0
        self.batch = None
        self.num_batches = 0

    def inputs(self):
        return self.batch[:-1]

    def targets(self):
        # targets are shifted by one
        return self.batch[1:]

    def init(self, filename, B, T):
        self.B = B;
        self.T = T;

        # open the input file for reading
        self.tokens_file = open(filename, "rb");
        self.tokens_file.seek(0, os.SEEK_END)
        self.file_size = self.tokens_file.tell()
        self.tokens_file.seek(0, os.SEEK_SET)
        if self.file_size < (B * T + 1) * 4:
            raise RuntimeError("Error: file size is too small for the batch size and sequence length");
        self.current_position = 0

        # allocate space for B*T + 1 integers to store the inputs and targets
        # Using CUDA CPU pinned memory for faster PCI Express transfers to GPU
        # See: https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/
        self.batch = cupyx.empty_pinned(B * T + 1, dtype=cupy.int32)
        self.num_batches = self.file_size // (B * T * 4);

    def reset(self):
        self.current_position = 0 

    def next_batch(self):
        B = self.B
        T = self.T
        # if we are at the end of the file, loop back to the beginning
        if self.current_position + (B*T+1) * 4 > self.file_size:
            self.current_position = 0;

        # read the B*T+1 integers from the file into batch
        self.tokens_file.seek(self.current_position, os.SEEK_SET)
        batch_fmt = f'{B*T+1}i'
        batch_len = struct.calcsize(batch_fmt)
        batch_unpack = struct.Struct(batch_fmt).unpack_from
        batch = batch_unpack(self.tokens_file.read(batch_len))
        self.batch[:] = batch 
        self.current_position += batch_len;

 
class Tokenizer:
    def __init__(self):
        self.vocab_size = 0
        self.token_table = None
        self.init_ok = 0

    def init(self, filename):
        with open(filename, "rb") as f:
            # if (file == NULL) {
            #     // try to be more helpful as we just added this feature, erase later
            #     printf("---\n");
            #     printf("WARNING: Failed to open the tokenizer file %s\n", filename);
            #     printf("The Tokenizer is a new feature added April 14 2024.\n");
            #     printf("Re-run `python train_gpt2.py` to write it\n");
            #     printf("---\n");
            #     tokenizer->init_ok = 0;
            #     return;
            # }
            header_fmt = '256i'
            header_len = struct.calcsize(header_fmt)
            header_unpack = struct.Struct(header_fmt).unpack_from
            header = header_unpack(f.read(header_len))
            assert(header[0] == 20240328)
            assert(header[1] == 1)
            self.vocab_size = header[2]
            # TODO: check if this DS is ok
            self.token_table = []
            for i in range(self.vocab_size):
                length = f.read(1)[0]
                token_bytes = f.read(length)
                self.token_table.append(token_bytes)
            selfinit_ok = 1;


NUM_PARAMETER_TENSORS = 16
       
class ParameterTensors:
    def __init__(self):
        self.wte = None  # (V, C)
        self.wpe = None  # (maxT, C)
        self.ln1w = None  # (L, C)
        self.ln1b = None  # (L, C)
        self.qkvw = None  # (L, 3*C, C)
        self.qkvb = None  # (L, 3*C)
        self.attprojw = None  # (L, C, C)
        self.attprojb = None  # (L, C)
        self.ln2w = None  # (L, C)
        self.ln2b = None  # (L, C)
        self.fcw = None  # (L, 4*C, C)
        self.fcb = None  # (L, 4*C)
        self.fcprojw = None  # (L, C, 4*C)
        self.fcprojb = None  # (L, C)
        self.lnfw = None  # (C)
        self.lnfb = None  # (C)

        # Used for iterate the tensors in order
        self.names = [
            "wte", "wpe", "ln1w", "ln1b", "qkvw", "qkvb", "attprojw", "attprojb",
            "ln2w", "ln2b", "fcw", "fcb", "fcprojw", "fcprojb", "lnfw", "lnfb"
        ]


class GPT2Config:
    def __init__(self):
        self.max_seq_len = None
        self.vocab_size = None
        self.num_layers = None
        self.num_heads = None
        self.channels = None

    def clone(self):
        new = GPT2Config()
        new.max_seq_len = self.max_seq_len
        new.vocab_size = self.vocab_size
        new.num_layers = self.num_layers
        new.num_heads = self.num_heads
        new.channels = self.channels
        return new

def malloc_and_point_parameters(params, param_sizes, num_parameters):
    # TODO(check): again, we are relying on cupy memory pool
    params_memory = cupy.empty(num_parameters, dtype=cupy.float32)
    # assign all the tensors their place in the array
    #next_param_ptr = params_memory.data.ptr
    #size_of_float = cupy.dtype(cupy.float32).itemsize
    #for i, n in enumerate(params.names):
    #    setattr(params, n, next_param_ptr)
    #    next_param_ptr += param_sizes[i] * size_of_float

    # Use a cupy view instead of a raw pointer
    current_size = 0
    for i, n in enumerate(params.names):
        setattr(params, n, params_memory[current_size:current_size+param_sizes[i]])
        current_size += param_sizes[i]

    return params_memory


NUM_ACTIVATION_TENSORS = 25

class ActivationTensors:
    def __init__(self):
        encoded = None  # (B, T, C)
        ln1 = None  # (L, B, T, C)
        ln1_mean = None  # (L, B, T)
        ln1_rstd = None  # (L, B, T)
        qkv = None  # (L, B, T, 3*C)
        atty = None  # (L, B, T, C)
        preatt = None  # (L, B, NH, T, T)
        att = None  # (L, B, NH, T, T)
        attproj = None  # (L, B, T, C)
        residual2 = None  # (L, B, T, C)
        ln2 = None  # (L, B, T, C)
        ln2_mean = None  # (L, B, T)
        ln2_rstd = None  # (L, B, T)
        fch = None  # (L, B, T, 4*C)
        fch_gelu = None  # (L, B, T, 4*C)
        fcproj = None  # (L, B, T, C)
        residual3 = None  # (L, B, T, C)
        lnf = None  # (B, T, C)
        lnf_mean = None  # (B, T)
        lnf_rstd = None  # (B, T)
        # if we have targets, this will be the logit _gradients_.
        logits = None  # (B, T, V)
        probs = None  # (B, T, V)
        losses = None  # (B, T)
        # adding these two compared to the CPU .c code, needed for attention kernel as buffers
        qkvr = None  # (L, B, T, 3*C)
        v_accum = None  # (L, B, T, C)

        self.names = [
            "encoded", "ln1", "ln1_mean", "ln1_rstd", "qkv", "atty", "preatt", "att", "attproj", "residual2",
            "ln2", "ln2_mean", "ln2_rstd", "fch", "fch_gelu", "fcproj", "residual3", "lnf", "lnf_mean", "lnf_rstd",
            "logits", "probs", "losses", "qkvr", "v_accum",
        ]



# Used for fwd and bwd
def fill_in_activation_sizes(act_sizes, B, T, config):
    V = config.vocab_size
    L = config.num_layers
    NH = config.num_heads
    C = config.channels
    act_sizes[0] = B * T * C  # encoded
    act_sizes[1] = L * B * T * C  # ln1
    act_sizes[2] = L * B * T  # ln1_mean
    act_sizes[3] = L * B * T  # ln1_rstd
    act_sizes[4] = L * B * T * 3*C  # qkv
    act_sizes[5] = L * B * T * C  # atty
    act_sizes[6] = B * NH * T * T  # preatt
    act_sizes[7] = L * B * NH * T * T  # att
    act_sizes[8] = L * B * T * C  # attproj
    act_sizes[9] = L * B * T * C  # residual2
    act_sizes[10] = L * B * T * C  # ln2
    act_sizes[11] = L * B * T  # ln2_mean
    act_sizes[12] = L * B * T  # ln2_rstd
    act_sizes[13] = L * B * T * 4*C  # fch
    act_sizes[14] = L * B * T * 4*C  # fch_gelu
    act_sizes[15] = L * B * T * C  # fcproj
    act_sizes[16] = L * B * T * C  # residual3
    act_sizes[17] = B * T * C  # lnf
    act_sizes[18] = B * T  # lnf_mean
    act_sizes[19] = B * T  # lnf_rstd
    act_sizes[20] = B * T * V  # logits
    act_sizes[21] = B * T * V  # probs
    act_sizes[22] = B * T  # losses
    act_sizes[23] = L * B * T * 3*C  # qkvr
    act_sizes[24] = B * T * C  # v_accum


# TODO Unify with the params code, is exactly the same
def malloc_and_point_activations(acts, act_sizes, num_activations):
    # TODO(check): again, we are relying on cupy memory pool
    acts_memory = cupy.empty(num_activations, dtype=cupy.float32)
    current_size = 0
    for i, n in enumerate(acts.names):
        setattr(acts, n, acts_memory[current_size:current_size+act_sizes[i]])
        current_size += act_sizes[i]

    return acts_memory


class GPT2:
    def __init__(self):
        # We just replicate the c++ structure, we could use cupy tensors & views for this
        # Each of the parameters is just a pointer to a big memory allocation
        self.params = ParameterTensors()
        self.params_sizes = [0 for i in range(NUM_PARAMETER_TENSORS)]
        self.params_memory = None
        self.config = GPT2Config()
        self.acts = ActivationTensors()
        self.acts_memory = None
        self.act_sizes = [0 for i in range(NUM_ACTIVATION_TENSORS)]
        self.grads = ParameterTensors()
        self.grads_memory = None
        self.grads_acts = ActivationTensors()
        self.grads_acts_memory = None
        self.num_parameters = 0
        self.m_memory = None
        self.v_memory = None

    def fill_in_parameter_sizes(self):
        V = self.config.vocab_size
        C = self.config.channels
        maxT = self.config.max_seq_len
        L = self.config.num_layers
        self.params_sizes[0] = V * C
        self.params_sizes[1] = maxT * C
        self.params_sizes[2] = L * C
        self.params_sizes[3] = L * C
        self.params_sizes[4] = L * (3 * C) * C
        self.params_sizes[5] = L * (3 * C)
        self.params_sizes[6] = L * C * C
        self.params_sizes[7] = L * C
        self.params_sizes[8] = L * C
        self.params_sizes[9] = L * C
        self.params_sizes[10] = L * (4 * C) * C
        self.params_sizes[11] = L * (4 * C)
        self.params_sizes[12] = L * C * (4 * C)
        self.params_sizes[13] = L * C
        self.params_sizes[14] = C
        self.params_sizes[15] = C

    def build_from_checkpoint(self, checkpoint_path):
        with open(checkpoint_path, 'rb') as f:
            header_fmt = '256i'
            header_len = struct.calcsize(header_fmt)
            header_unpack = struct.Struct(header_fmt).unpack_from
            model_header = header_unpack(f.read(header_len))
            if model_header[0] != 20240326:
                 raise RuntimeError("Bad magic model file")
            if model_header[1] != 1:
                 raise RuntimeError("Bad version in model file")

            self.config.max_seq_len = maxT = model_header[2];
            self.config.vocab_size = V = model_header[3];
            self.config.num_layers = L = model_header[4];
            self.config.num_heads = NH = model_header[5];
            self.config.channels = C = model_header[6];
            print("[GPT-2]");
            print(f"max_seq_len: {maxT}");
            print(f"vocab_size: {V}");
            print(f"num_layers: {L}");
            print(f"num_heads: {NH}");
            print(f"channels: {C}");
            self.fill_in_parameter_sizes()

            num_parameters = 0;
            for i in range(NUM_PARAMETER_TENSORS):
                num_parameters += self.params_sizes[i]

            print(f"num_parameters: {num_parameters}");
            self.num_parameters = num_parameters
       
            # create memory for model parameters on the device
            # NOTE: need the function to accept parameters due to grad initialization
            self.params_memory = malloc_and_point_parameters(self.params, self.params_sizes, num_parameters)
            size_in_mb = int(round(num_parameters * cupy.dtype(cupy.float32).itemsize) / (1024*1024))
            print(f"allocated {size_in_mb} MiB for model parameters")
      
            # Read the parameters and copy them to a numpy array
            params_fmt = f'{num_parameters}f'
            params_len = struct.calcsize(params_fmt)
            params_unpack = struct.Struct(params_fmt).unpack_from
            params_memory_cpu = numpy.array(params_unpack(f.read(params_len)), dtype=numpy.float32)
            # NOTE: we could just allocate the cupy array here directly with a constructor
            # copyto & [:] = numpy doesnt work due to cupy not liking implicit data transferences
            cupy.cuda.runtime.memcpy(
                self.params_memory.data.ptr, params_memory_cpu.ctypes.data,
                params_memory_cpu.nbytes, cupy.cuda.runtime.memcpyHostToDevice
            )
        # other inits
        self.batch_size = 0
        self.seq_len = 0
        self.mean_loss = -1.0  # -1.0f will designate no loss

    def forward(self, inputs, targets, B, T):
        # ensure the model was initialized or error out
        if self.params_memory is None:
            raise RuntimeError("Error: model was not initialized properly.\n")

        # convenience parameters
        V = self.config.vocab_size
        L = self.config.num_layers
        NH = self.config.num_heads
        C = self.config.channels
    
        # Validate inputs, all indices must be in the range [0, V)
        for i in range(B * T):
             assert(0 <= inputs[i] < V)
             if targets is not None:
                 assert(0 <= targets[i] < V)
    
        # allocate space for all the activations if needed (done here, lazily)
        if self.acts_memory is None:
            # record the current B,T as well
            self.batch_size = B
            self.seq_len = T
            # and now allocate the space
            fill_in_activation_sizes(self.act_sizes, B, T, self.config);
            num_activations = 0
            for i in range(NUM_ACTIVATION_TENSORS):
                num_activations += self.act_sizes[i]
            print("num_activations", num_activations)         
            self.num_activations = num_activations
            self.acts_memory = malloc_and_point_activations(self.acts, self.act_sizes, num_activations);
            acts_memory = int(round(num_activations * cupy.dtype(cupy.float32).itemsize / (1024 * 1024)))
            print(f"allocated {acts_memory} MiB for activations")
            # Regular memory allocations
            # self.inputs = cupy.empty((B * T), dtype=cupy.int32);
            # self.target = cupy.empty((B * T), dtype=cupy.int32);
            # self.cpu_losses = cupy.empty((B * T), dtype=cupy.float32);
            self.cpu_losses = cupyx.empty_pinned(B * T, dtype=cupy.float32)
        else:
            # validate B,T is consistent with how we've allocated the memory before
            # in principle we could get more clever here in the future, for now this is safest
            if (B != self.batch_size or T != self.seq_len):
                raise RuntimeError(f"Model: B={self.batch_size} T={self.seq_len}, Desired: B={B} T={T}")

        # copy inputs/targets to the model
        # We are just creating a new cupy array time, memory is drawn from the pool and the numpy array
        # with the inputs is in pinned memory
        self.inputs = cupy.array(inputs)
        if targets is not None:
            self.targets = cupy.array(targets)

        # forward pass
        params = self.params
        acts = self.acts
        encoder_forward(acts.encoded, self.inputs, params.wte, params.wpe, B, T, C, V)
        compare_tensor(acts.encoded, "acts.encoded", B * T * C)
        for l in range(L):

            residual = acts.encoded if l == 0 else acts.residual3[(l-1) * B * T * C:]

            l_ln1w = params.ln1w [l * C:]
            l_ln1b = params.ln1b [l * C:]
            l_qkvw = params.qkvw [l * 3*C * C:]
            l_qkvb = params.qkvb [l * 3*C:]
            l_attprojw = params.attprojw[l * C * C:]
            l_attprojb = params.attprojb[l * C:]
            l_ln2w = params.ln2w[l * C:]
            l_ln2b = params.ln2b[l * C:]
            l_fcw = params.fcw[l * 4*C * C:]
            l_fcb = params.fcb[l * 4*C:]
            l_fcprojw = params.fcprojw[l * C * 4*C:]
            l_fcprojb = params.fcprojb[l * C:]

            l_ln1 = acts.ln1[l * B * T * C:]
            l_ln1_mean = acts.ln1_mean[l * B * T:]
            l_ln1_rstd = acts.ln1_rstd[l * B * T:]
            l_qkv = acts.qkv[l * B * T * 3*C:]
            l_qkvr = acts.qkvr[l * B * T * 3*C:]
            l_atty = acts.atty[l * B * T * C:]
            l_att = acts.att[l * B * NH * T * T:]
            l_attproj = acts.attproj[l * B * T * C:]
            l_residual2 = acts.residual2[l * B * T * C:]
            l_ln2 = acts.ln2[l * B * T * C:]
            l_ln2_mean = acts.ln2_mean[l * B * T:]
            l_ln2_rstd = acts.ln2_rstd[l * B * T:]
            l_fch = acts.fch[l * B * T * 4*C:]
            l_fch_gelu = acts.fch_gelu[l * B * T * 4*C:]
            l_fcproj = acts.fcproj[l * B * T * C:]
            l_residual3 = acts.residual3[l * B * T * C:]
            # these are only needed as scratchpads for the forward pass, but
            # need not be stored for backward
            l_preatt = acts.preatt
            l_v_accum = acts.v_accum

            layernorm_forward(l_ln1, l_ln1_mean, l_ln1_rstd, residual, l_ln1w, l_ln1b, B, T, C)
            compare_tensor(l_ln1, f"{l}_l_ln1", B * T * C)
            compare_tensor(l_ln1_mean, f"{l}_l_ln1_mean", B * T)
            compare_tensor(l_ln1_rstd, f"{l}_l_ln1_rstd", B * T)
            matmul_forward_cublaslt(l_qkv, l_ln1, l_qkvw, l_qkvb, B, T, C, 3*C)
            compare_tensor(l_qkv, f"{l}_l_qkv", B * T * 3 * C);
            attention_forward(l_atty, l_v_accum, l_qkvr, l_preatt, l_att, l_qkv, B, T, C, NH)
            compare_tensor(l_atty, f"{l}_l_atty", B * T * C);
            compare_tensor(l_v_accum, f"{l}_l_v_accum", B * T * C);
            compare_tensor(l_qkvr, f"{l}_l_qkvr", B * T * 3 * C);
            compare_tensor(l_preatt, f"{l}_l_preatt", B * NH * T * T);
            compare_tensor(l_att, f"{l}_l_att", B * NH * T * T);
            matmul_forward_cublaslt(l_attproj, l_atty, l_attprojw, l_attprojb, B, T, C, C)
            compare_tensor(l_attproj, f"{l}_l_attproj", B * T * C)
            residual_forward(l_residual2, residual, l_attproj, B*T*C)
            compare_tensor(l_residual2, f"{l}_l_residual2", B * T * C)
            layernorm_forward(l_ln2, l_ln2_mean, l_ln2_rstd, l_residual2, l_ln2w, l_ln2b, B, T, C)
            matmul_forward_cublaslt(l_fch, l_ln2, l_fcw, l_fcb, B, T, C, 4*C)
            gelu_forward(l_fch_gelu, l_fch, B*T*4*C)
            matmul_forward_cublaslt(l_fcproj, l_fch_gelu, l_fcprojw, l_fcprojb, B, T, 4*C, C)
            residual_forward(l_residual3, l_residual2, l_fcproj, B*T*C)

        residual = acts.residual3[(L-1) * B * T * C:] # last residual is in residual3
        layernorm_forward(acts.lnf, acts.lnf_mean, acts.lnf_rstd, residual, params.lnfw, params.lnfb, B, T, C)
        matmul_forward_cublas(acts.logits, acts.lnf, params.wte, None, B, T, C, V)
        if targets is not None:
            fused_classifier3(acts.logits, acts.losses, self.targets.ravel(), B, T, V, V)
            self.cpu_losses = cupy.asnumpy(acts.losses[:B * T])
            mean_loss = numpy.mean(self.cpu_losses)
            self.mean_loss = mean_loss;
            #print(mean_loss)
        else:
            pass

    def zero_grad(self):
        if self.grads_memory is not None:
            self.grads_acts_memory.fill(0.0)
            self.grads_memory.fill(0.0)

    def backward(self):
        if self.mean_loss == -1.0:
            raise RuntimeError("Error: must forward with targets before backward")

        if self.grads_memory is None:
            self.grads_memory = malloc_and_point_parameters(self.grads, self.params_sizes, self.num_parameters);
            size_in_mb = int(round(self.num_parameters * cupy.dtype(cupy.float32).itemsize) / (1024*1024))
            print(f"allocated {size_in_mb} MiB for parameter gradients")
            # we're going to be clever for the activations backward pass. we don't need to exactly
            # mirror the forward pass acrtivations and we will save memory.
            bw_act_sizes = [0] * NUM_ACTIVATION_TENSORS
            cfg = self.config.clone()
            cfg.num_layers = 1; # copy the configuration but override number of layers to 1
            fill_in_activation_sizes(bw_act_sizes, self.batch_size, self.seq_len, cfg);
            # on top of that, some buffers are not needed at all, set their sizes to zero
            bw_act_sizes[0] = 0  #  encoded
            bw_act_sizes[2] = 0  #  ln1_mean
            bw_act_sizes[3] = 0  #  ln1_rstd
            bw_act_sizes[8] = 0  #  attproj
            bw_act_sizes[9] = 0  #  residual2
            bw_act_sizes[11] = 0  # ln2_mean
            bw_act_sizes[12] = 0  # ln2_rstd
            bw_act_sizes[18] = 0  # lnf_mean
            bw_act_sizes[19] = 0  # lnf_rstd
            bw_act_sizes[21] = 0  # probs
            # count up and allocate the space
            self.num_grad_acts = 0;
            for i in range(NUM_ACTIVATION_TENSORS):
                self.num_grad_acts += bw_act_sizes[i];
            self.grads_acts_memory = malloc_and_point_activations(self.grads_acts, bw_act_sizes, self.num_grad_acts);

            size_in_mb = int(round(self.num_grad_acts * cupy.dtype(cupy.float32).itemsize) / (1024*1024))
            print(f"allocated {size_in_mb} MiB for activation gradients")
            # init gradients of parameters and activations to zero
            self.zero_grad();

            # convenience shortcuts
            B = self.batch_size
            T = self.seq_len
            V = self.config.vocab_size
            L = self.config.num_layers
            NH = self.config.num_heads
            C = self.config.channels
        
            # backward pass: go in the reverse order of the forward pass, and call backward() functions
            params = self.params  # for brevity
            grads = self.grads
            acts = self.acts
            grads_acts = self.grads_acts

            # we kick off the chain rule by filling in dlosses with 1.0f/(B*T)
            # this was done in the fused classifier kernel as last step of forward pass
            # technically that is a small, inline backward() pass of calculating
            # total, final loss as the mean over all losses over all (B,T) positions in the batch
            # next: backward the classifier matmul
            matmul_backward(grads_acts.lnf, grads.wte, None, acts.logits, acts.lnf, params.wte, B, T, C, V)
            #print4(grads_acts.lnf)
            #print4(grads.wte)
            compare_tensor(acts.lnf, "acts.lnf", B * T * C);
            compare_tensor(params.wte, "params.wte", V * C);
            compare_tensor(grads_acts.lnf, "grad_acts.lnf", B * T * C);
            compare_tensor(grads.wte, "grads.wte", V * C);
            #sys.exit(0)
            residual = acts.residual3[(L-1) * B * T * C:]  # last residual is in residual3
            dresidual = grads_acts.residual3  # the main buffer holding the gradient in the backward pass
            layernorm_backward(dresidual, grads.lnfw, grads.lnfb, grads_acts.lnf, residual, params.lnfw, acts.lnf_mean, acts.lnf_rstd, B, T, C)
            #print4(dresidual)
            #print4(grads.lnfw)
            #print4(grads.lnfb)
            # now backward all the layers
            for l in range(L-1, -1, -1):
                residual = acts.encoded if l == 0 else acts.residual3[(l-1) * B * T * C:]
                # get the pointers of the weights for this layer
                l_ln1w = params.ln1w[l * C:]
                l_qkvw = params.qkvw[l * 3*C * C:]
                l_attprojw = params.attprojw [l * C * C:]
                l_ln2w = params.ln2w[l * C:]
                l_fcw = params.fcw[l * 4*C * C:]
                l_fcprojw = params.fcprojw[l * C * 4*C:]
                # get the pointers of the gradients of the weights for this layer
                dl_ln1w = grads.ln1w[l * C:]
                dl_ln1b = grads.ln1b[l * C:]
                dl_qkvw = grads.qkvw[l * 3*C * C:]
                dl_qkvb = grads.qkvb[l * 3*C:]
                dl_attprojw = grads.attprojw[l * C * C:]
                dl_attprojb = grads.attprojb[l * C:]
                dl_ln2w = grads.ln2w[l * C:]
                dl_ln2b = grads.ln2b[l * C:]
                dl_fcw = grads.fcw[l * 4*C * C:]
                dl_fcb = grads.fcb[l * 4*C:]
                dl_fcprojw = grads.fcprojw[l * C * 4*C:]
                dl_fcprojb = grads.fcprojb[l * C:]
                # get the pointers of the activations for this layer
                l_ln1 = acts.ln1[l * B * T * C:]
                l_ln1_mean = acts.ln1_mean[l * B * T:]
                l_ln1_rstd = acts.ln1_rstd[l * B * T:]
                l_qkv = acts.qkv[l * B * T * 3*C:]
                l_qkvr = acts.qkvr[l * B * T * 3*C:]
                l_atty = acts.atty[l * B * T * C:]
                l_att = acts.att[l * B * NH * T * T:]
                l_residual2 = acts.residual2[l * B * T * C:]
                l_ln2 = acts.ln2[l * B * T * C:]
                l_ln2_mean = acts.ln2_mean[l * B * T:]
                l_ln2_rstd = acts.ln2_rstd[l * B * T:]
                l_fch = acts.fch[l * B * T * 4*C:]
                l_fch_gelu = acts.fch_gelu[l * B * T * 4*C:]
                # get the pointers of the gradients of the activations for this layer
                # notice that there is no l *, because we just have a single copy, and keep
                # re-using this memory in every Transformer block as we calculate backward pass
                dl_ln1 = grads_acts.ln1
                dl_qkv = grads_acts.qkv
                dl_qkvr = grads_acts.qkvr
                dl_atty = grads_acts.atty
                dl_preatt = grads_acts.preatt
                dl_att = grads_acts.att
                dl_v_accum = grads_acts.v_accum
                dl_ln2 = grads_acts.ln2
                dl_fch = grads_acts.fch
                dl_fch_gelu = grads_acts.fch_gelu
                matmul_backward(dl_fch_gelu, dl_fcprojw, dl_fcprojb, dresidual, l_fch_gelu, l_fcprojw, B, T, 4*C, C)
                gelu_backward(dl_fch, l_fch, dl_fch_gelu, B*T*4*C)
                matmul_backward(dl_ln2, dl_fcw, dl_fcb, dl_fch, l_ln2, l_fcw, B, T, C, 4*C)
                # layernorm backward does += to the dresidual, so it correctly accumulates grad from the MLP block above
                layernorm_backward(dresidual, dl_ln2w, dl_ln2b, dl_ln2, l_residual2, l_ln2w, l_ln2_mean, l_ln2_rstd, B, T, C)
                matmul_backward(dl_atty, dl_attprojw, dl_attprojb, dresidual, l_atty, l_attprojw, B, T, C, C)
                #print4(dl_atty)
                #print4(dl_attprojw)
                #print4(dl_attprojb)
                attention_backward(dl_qkv, dl_qkvr, dl_preatt, dl_att, dl_v_accum, dl_atty, l_qkv, l_qkvr, l_att, B, T, C, NH)
                #print4(dl_qkv);
                #print4(dl_qkvr);
                #print4(dl_preatt);
                #print4(dl_att);
                #print4(dl_v_accum);
                #print4(dl_atty);
                matmul_backward(dl_ln1, dl_qkvw, dl_qkvb, dl_qkv, l_ln1, l_qkvw, B, T, C, 3*C)
                # layernorm backward does += to dresidual, so it correctly accumulates gradient for the Attention block above
                layernorm_backward(dresidual, dl_ln1w, dl_ln1b, dl_ln1, residual, l_ln1w, l_ln1_mean, l_ln1_rstd, B, T, C);
                #print4(dresidual)
            encoder_backward(grads.wte, grads.wpe, dresidual, self.inputs, B, T, C)

    def update(self, learning_rate, beta1, beta2, eps, weight_decay, t):
        # reference: https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html

        # lazily allocate the memory for m_memory and v_memory
        if self.m_memory is None:
            self.m_memory = cupy.zeros(self.num_parameters, dtype=cupy.float32)
            self.v_memory = cupy.zeros(self.num_parameters, dtype=cupy.float32)
            size_in_mb = int(round(self.num_parameters * cupy.dtype(cupy.float32).itemsize) / (1024*1024))
            print(f"allocated {size_in_mb} MiB for AdamW optimizer state m")
            print(f"allocated {size_in_mb} MiB for AdamW optimizer state v")

        block_size = 512
        num_blocks = ceil_div(self.num_parameters, block_size)
        beta1_correction = 1.0 - math.pow(beta1, t)
        beta2_correction = 1.0 - math.pow(beta2, t)

        adamw_kernel2[num_blocks, block_size](
            self.params_memory,
            self.grads_memory,
            self.m_memory,
            self.v_memory,
            self.num_parameters,
            learning_rate, beta1, beta2, beta1_correction, beta2_correction, eps, weight_decay)


def main():
    global cublaslt_workspace_size
    global cublaslt_workspace
    global cublas_compute_type
    global cublas_handle
    global cublaslt_handle

    # Default values
    input_dataset_prefix = "data/tiny_shakespeare"
    output_log_file = None
    B = 4
    T = 1024
    learning_rate = 1e-4
    val_loss_every = 20
    val_max_batches = 20
    sample_every = 20
    genT = 64
    # TODO override avove with argparse
    print(f"input dataset prefix: {input_dataset_prefix}")
    print(f"output log file: {'NULL' if output_log_file is None else output_log_file}")
    print(f"batch size B: {B}")
    print(f"sequence length T: {T}")
    print(f"learning rate: {learning_rate}")
    print(f"val_loss_every: {val_loss_every}")
    print(f"val_max_batches: {val_max_batches}")
    print(f"sample_every: {sample_every}")
    print(f"genT: {genT}")

    # set up the device
    deviceIdx = 0
    checkCudaErrors(cudart.cudaSetDevice(deviceIdx))    
    deviceProp = checkCudaErrors(cudart.cudaGetDeviceProperties(deviceIdx))
    print("[System]")
    print(f"Device {deviceIdx}: {deviceProp.name.decode('utf-8')}")
    # For nvmath errors are thrown in a pythonic way
    cublas_handle = cublas.create()
    cublaslt_handle = cublaslt.create()
    enable_tf32 = deviceProp.major >= 8
    print(f"enable_tf32: {enable_tf32}")
    if enable_tf32: 
        cublas_compute_type =  cublas.ComputeType.COMPUTE_32F_FAST_TF32
        cublas_math_mode = cublas.Math.TF32_TENSOR_OP_MATH
    else:
        cublas_compute_type =  cublas.ComputeType.COMPUTE_32F
        cublas_math_mode = cublas.Math.DEFAULT_MATH

    cublas.set_math_mode(cublas_handle, cublas_math_mode)
    # setup the (global) cuBLASLt workspace
    # TODO(check): this inits cupy memory pool, it may create conflicts with other libraries memory allocation
    cublaslt_workspace = cupy.empty(cublaslt_workspace_size, dtype=cupy.uint8)
    #thrust::device_vector<std::uint8_t> cublaslt_workspace_vec(cublaslt_workspace_size)
    #cublaslt_workspace = thrust::raw_pointer_cast(cublaslt_workspace_vec.data())
    model = GPT2()
    model.build_from_checkpoint("gpt2_124M.bin")

    train_tokens_filename = f"{input_dataset_prefix}_train.bin"
    val_tokens_filename = f"{input_dataset_prefix}_val.bin"

    train_loader = DataLoader()
    train_loader.init(train_tokens_filename, B, T);
    val_loader = DataLoader()
    val_loader.init(val_tokens_filename, B, T);

    train_num_batches = train_loader.num_batches  # let's do 1 epoch by default
    val_num_batches = train_loader.num_batches if train_loader.num_batches < val_max_batches else val_max_batches;
    print(f"train dataset num_batches: {train_loader.num_batches}", train_loader.num_batches);
    print(f"val dataset num_batches: {val_loader.num_batches}", val_loader.num_batches);

    # TODO logger


    tokenizer = Tokenizer()
    tokenizer.init("gpt2_tokenizer.bin")

    for step in range(train_num_batches):
        last_step = step == train_num_batches
        # once in a while estimate the validation loss
        if step % val_loss_every == 0 or last_step:
            val_loss = 0.0
            val_loader.reset()
            for i in range(val_num_batches):
                val_loader.next_batch()
                model.forward(val_loader.inputs(), val_loader.targets(), B, T)
                val_loss += model.mean_loss
            val_loss /= val_num_batches;
            print(f"val loss {val_loss}");
            #logger_log_val(&logger, step, val_loss);

        if last_step:
            break
        start = time.time()
        train_loader.next_batch()
 
        model.forward(train_loader.inputs(), train_loader.targets(), B, T)
        model.zero_grad()
        model.backward()
        model.update(learning_rate, 0.9, 0.999, 1e-8, 0.0, step+1)
        print(f"step {step+1}/{train_num_batches}: train loss {model.mean_loss} ({int((time.time() - start)*1000)} ms)")

if __name__ == "__main__":
    main()
