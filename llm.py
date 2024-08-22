import os
import struct

import cupy
import cupyx
import numpy

from cuda import cuda, cudart, nvrtc
from nvmath.bindings import cublas, cublasLt


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
        setattr(params, n, params_memory[current_size:])
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
        setattr(acts, n, acts_memory[current_size:])
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
            self.inputs = cupy.empty((B * T), dtype=cupy.int32);
            self.target = cupy.empty((B * T), dtype=cupy.int32);
            self.cpu_losses = cupy.empty((B * T), dtype=cupy.float32);
        else:
            # validate B,T is consistent with how we've allocated the memory before
            # in principle we could get more clever here in the future, for now this is safest
            if (B != self.batch_size or T != self.seq_len):
                raise RuntimeError(f"Model: B={self.batch_size} T={self.seq_len}, Desired: B={B} T={T}")



def main():
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
    cublaslt_handle = cublasLt.create()
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
                print(val_loader.inputs())
                print(val_loader.targets())
                model.forward(val_loader.inputs(), val_loader.targets(), B, T)
                sys.exit(0)
                #val_loss += model.mean_loss
            val_loss /= val_num_batches;
            print(f"val loss {val_loss}");
            #logger_log_val(&logger, step, val_loss);


if __name__ == "__main__":
    main()
