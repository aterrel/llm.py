import math
import os
import struct
import sys

import cupy
import cupyx
import numba.cuda
import numpy

from cuda import cuda, cudart, nvrtc
from nvmath.bindings import cublas
from nvmath.bindings import cublasLt as cublaslt


cublaslt_workspace_size = 32 * 1024 * 1024
cublaslt_workspace = None
cublas_compute_type = None
cublas_handle = None
cublaslt_handle = None


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
--------------------- KERNELS ---------------------
"""

@numba.cuda.jit
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
    sums[numba.cuda.threadIdx.x] = 0
    x = inp[idx * C:]
    for i in range(thread_rank, C, warp_size):
        sums[numba.cuda.threadIdx.x] += x[i]

    # TODO support CG Reduction, right now we do it naively
    numba.cuda.syncthreads()
    if (thread_rank == 0):
        for i in range(meta_group_rank * warp_size + 1, meta_group_rank * warp_size + warp_size):
            sums[meta_group_rank * warp_size] +=  sums[i]
    numba.cuda.syncthreads()
    m = sums[meta_group_rank * warp_size] / C
    if (thread_rank == 0):
        mean[idx] = m

    # rstd
    sums[numba.cuda.threadIdx.x] = 0
    for i in range(thread_rank, C, warp_size):
        diff = x[i] - m
        sums[numba.cuda.threadIdx.x] += diff * diff

    numba.cuda.syncthreads()
    if (thread_rank == 0):
        for i in range(meta_group_rank * warp_size + 1, meta_group_rank * warp_size + warp_size):
            sums[meta_group_rank * warp_size] +=  sums[i]
    numba.cuda.syncthreads()
      
    s = sums[meta_group_rank * warp_size]
    s = 1.0/ math.sqrt(s / C + 1e-5)

    if (thread_rank == 0):
        rstd[idx] = s

    # final normalization and scaling by weight/bias
    o = out[idx * C:]
    for c in range(thread_rank, C, warp_size):
        n = s  * x[c] - m
        o[c] = n *weight[c] + bias[c]


def layernorm_forward(out, mean,  rstd, inp, weight, bias, B, T, C):
    def ceil_div(a, b):
        return -(a // -b)

    block_size = 512
    N = B * T
    grid_size = ceil_div(N * 32, block_size)
    layernorm_forward_kernel3[grid_size, block_size](out, mean, rstd, inp, weight, bias, N, C)


def cublaslt_setattr(matmul_desc, name, value):
    name = name.upper()
    DescEnum = cublaslt.MatmulDescAttribute 
    scalar_attrs =  [e.name for e in DescEnum] 
    if name not in scalar_attrs:
        return 
    get_dtype = cublaslt.get_matmul_desc_attribute_dtype
    attribute_buffer = numpy.zeros((1,), dtype=get_dtype(DescEnum[name]))
    attribute_buffer[0] = value
    cublaslt.matmul_desc_set_attribute(matmul_desc, DescEnum[name].value, attribute_buffer.ctypes.data, attribute_buffer.itemsize)

def cublaslt_set_preference_attr(preference, name, value):
    name = name.upper()
    PreferenceEnum = cublaslt.MatmulPreferenceAttribute
    scalar_attrs =  [e.name for e in PreferenceEnum] 
    if name not in scalar_attrs:
        return 
    get_dtype = cublaslt.get_matmul_preference_attribute_dtype
    attribute_buffer = numpy.zeros((1,), dtype=get_dtype(PreferenceEnum[name]))
    attribute_buffer[0] = value
    cublaslt.matmul_preference_set_attribute(preference, PreferenceEnum[name].value, attribute_buffer.ctypes.data, attribute_buffer.itemsize)


def matmul_forward_cublaslt(out, inp, weight, bias, B, T, C, OC):
    has_bias = (bias is not None);

    # check bias alignment
    if bias.data.ptr % 16 != 0:
        raise RuntimeError("Bias pointer is not aligned (cuBLASLt requirement)!\n")

    returnedResults = 0
    # create the operation descriptor
    opNoTranspose = cublas.Operation.N
    opTranspose = cublas.Operation.T
    epilogueBias = cublaslt.Epilogue.BIAS
    cuda_r_32f = cudart.cudaDataType.CUDA_R_32F
    operation_desc = cublaslt.matmul_desc_create(cublas_compute_type, cuda_r_32f)
    cublaslt_setattr(operation_desc, "TRANSA", opTranspose)
    cublaslt_setattr(operation_desc, "TRANSB", opNoTranspose)
    cublaslt_setattr(operation_desc, "EPILOGUE", epilogueBias)
    if has_bias:
        cublaslt_setattr(operation_desc, "BIAS_POINTER", bias.data.ptr)
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
    print("Algorithm")
    cublaslt.matmul_algo_get_heuristic(
        cublaslt_handle, operation_desc, weight_layout, input_layout, output_layout, output_layout, preference, 1, algorithms_buffer.ctypes.data, num_algorithms.ctypes.data
    )
    print("Algorithm ", num_algorithms)
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


@numba.cuda.jit(device=True)
def i2n(idx, E1, E2):
    bt = idx // E1
    b = bt // E2
    t = bt % E2
    c = idx % E1
    return (b, t, c)


@numba.cuda.jit
def encoder_forward_kernel(out_md, wte_md, wpe_md, inp_md, C, T):
    idx = numba.cuda.grid(1)
    # Ensure the thread is within the bounds of the array
    if idx < out_md.size:
        b, t, c = i2n(idx, C, T)
        out_md[b, t, c] = wte_md[inp_md[b, t], c] + wpe_md[t, c]
   

def encoder_forward(out, inpv, wte, wpe, B, T, C, V):
    out_md = out.reshape(B, T, C)
    wte_md = wte.reshape(V, C)
    wpe_md = wpe.reshape(T, C)
    inp_md = inpv.reshape(B, T)
    threads_per_block = 256  
    blocks_per_grid = (out.size + threads_per_block - 1) // threads_per_block
    encoder_forward_kernel[blocks_per_grid, threads_per_block](out_md, wte_md, wpe_md, inp_md, C, T)

"""
---------------------------------------------------
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

            layernorm_forward(l_ln1, l_ln1_mean, l_ln1_rstd, residual, l_ln1w, l_ln1b, B, T, C)
            #print(C, residual.shape, l_ln1_mean.shape)
            matmul_forward_cublaslt(l_qkv, l_ln1, l_qkvw, l_qkvb, B, T, C, 3*C)
            for i in range(B*T*C -50, B*T*C, 2):
                print(l_qkv[i], ' ', l_qkv[i+1])
            sys.exit(0)

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
                sys.exit(0)
                #val_loss += model.mean_loss
            val_loss /= val_num_batches;
            print(f"val loss {val_loss}");
            #logger_log_val(&logger, step, val_loss);


if __name__ == "__main__":
    main()
