# Quick Start - Metal

Apple does offer an [example](https://developer.apple.com/documentation/metal/basic_tasks_and_concepts/performing_calculations_on_a_gpu?preferredLanguage=occ) explaining how to use Metal performing a vector addition operation.

This quick start can be considered as a refactored version, leaving out unnecessary details if you are familiar with CUDA.

But I have to inform readers before we jump into this example. When using Metal as a backend for Machine Learning, we have to admit there are limited chances we will write the kernel directly. Apple also prepared a Metal Performance Shading library, which is closer to Machine Learning.

## Kernel

Let's see what the vect-add kernel function looks like in CUDA:

```CUDA
__global__ void vector_add_cuda(int *A, 
                                int *B, 
                                int *C)
{
   int index = blockIdx.x * blockDim.x + threadIdx.x;

   C[index] = A[index] + B[index]; 
}
```

And let's see what OpenCL:

```OpenCL
__kernel void vector_add_cl(__global const float *A, 
                            __global const float *B, 
                            __global float *restrict C)
{
    int index = get_global_id(0);

    C[index] = A[index] + B[index];
}
```

The Metal Shading Language version is a bit different version:

```MSL
kernel void add_arrays(device const float* A,
                       device const float* B,
                       device float* C,
                       uint index [[thread_position_in_grid]])
{
    C[index] = A[index] + B[index];
}
```

Ok, so basically, they are almost identical. The different in Metal Shading Language is that the `index` is passed as a argument.

Now, let's see how to compile this script and execute it on macOS.

## Context Setting

In CUDA, you may hear about something called context, though using CUDA runtime APIs permits developers from worrying about it.

To keep it brief, context is a packaged definition of your device, command queue to the device, data allocated through the command queue and modules loaded. In Metal, we still need to manage these objects manually.

### Device

Let's find the device first.

The simplest approach is to use [`MTLCreateSystemDefaultDevice`](https://developer.apple.com/documentation/metal/1433401-mtlcreatesystemdefaultdevice?language=objc).

```Objective-C
id<MTLDevice> device = MTLCreateSystemDefaultDevice();
```

The type of a Metal Device is named as [`MTLDevice`](https://developer.apple.com/documentation/metal/mtldevice?language=objc).

*All Metal APIs, including MPS APIs, starts with 3 all capital words: `MTL`.*

If there are multiple Metal devices attached to the machine, [`MTLCopyAllDevices`](https://developer.apple.com/documentation/metal/1433367-mtlcopyalldevices?language=objc) can help developers to get all of them.

### Command Queue

To create a [MTLCommandQueue](https://developer.apple.com/documentation/metal/mtlcommandqueue?language=objc), `MTLDevice` offers a method called [`newCommandQueue`](https://developer.apple.com/documentation/metal/mtldevice/1433388-newcommandqueue). This concept is also shared with CUDA and OpenCL.

## Data

Let's prepare data first.

### Allocation and Setting Values

If we want a fresh allocation of memory associated with the device, we can use method [`newBufferWithLength`](https://developer.apple.com/documentation/metal/mtldevice/1433375-newbufferwithlength?language=objc) from `MTLDeivce`.

If we already have a piece of data on the host, and wish to create a copy on a Metal device, [`newBufferWithBytes`](https://developer.apple.com/documentation/metal/mtldevice/1433429-newbufferwithbytes?language=objc) can help.

Both function returns the basic data representative: [`MTLBuffer`](https://developer.apple.com/documentation/metal/mtlbuffer?language=objc).

There are multiple storage options, you can check this [page](https://developer.apple.com/documentation/metal/mtlresourceoptions?language=objc) for more details.

## Function

### Compile Kernel

With the device object, we can compile the kernel function now. The kernel script written in Metal Shading Language is already saved to a file named `add.metal`.

[`newDefaultLibrary`](https://developer.apple.com/documentation/metal/mtldevice/1433380-newdefaultlibrary) is a method of `MTLDevice` and will search all the `.metal` file in this application and compile them into a single library. This is pretty similar to what we know as **online compiling** in CUDA/OpenCL. Of course, Metal also support **offline compiling**, which compiles the `.metal` files into one or multiple `.metallib` file(s) and loads a compiled library with `MTLDevice` method [`newLibraryWithFile`](https://developer.apple.com/documentation/metal/mtldevice/1433416-newlibrarywithfile?language=objc) with specified file path. There are also multiple methods for online and offline compiling, such as [`newlibrarywithsource`](https://developer.apple.com/documentation/metal/mtldevice/1433351-newlibrarywithsource?language=objc) and [`newlibrarywithdata`](https://developer.apple.com/documentation/metal/mtldevice/1433391-newlibrarywithdata?language=objc). 

However, let's face it, that for most cases in machine learning framework, there are limited chances we will write the kernel function directly. Upper stream developers, I mean those from NVIDIA/AMD/Apple has more advantages when writing these kernel functions, and they do want to complete the kernels for ML (well, at lease for NVIDIA).

The library we just loaded are defined as [`MTLLibrary`](https://developer.apple.com/documentation/metal/mtllibrary?language=objc).

To extract a [`MTLFunction`](https://developer.apple.com/documentation/metal/mtlfunction?language=objc) from a `MTLLibrary`, we just need fed the function name into [`newFunctionWithName`](https://developer.apple.com/documentation/metal/mtllibrary/1515524-newfunctionwithname?language=objc). *(Other MTLLibrary methods seem implying a template kernel function is possible in Metal Shading Language. I will take a further test and update this section later.)*

### Pipeline Status

According to the [example](https://developer.apple.com/documentation/metal/basic_tasks_and_concepts/performing_calculations_on_a_gpu?preferredLanguage=occ) from Apple, the function we just extracted is not executable code but a proxy. In Metal, [`MTLComputePipelineState`](https://developer.apple.com/documentation/metal/mtlcomputepipelinestate?language=objc) is needed to convert `MTLFunction` to `MTLComputePipelineState` with `MTLDevice` method [`newComputePipelineStateWithFunction`](https://developer.apple.com/documentation/metal/mtldevice/1433427-newcomputepipelinestatewithfunct?language=objc).

## Setup Function

Now we have the data and function both ready. All we need to do is to setup function with data and send it to command queue.

### Command Buffer

`MTLCommandQueue` only accepts [`MTLCommandBuffer`](https://developer.apple.com/documentation/metal/mtlcommandbuffer?language=objc). So let's create an empty command buffer.

```Objective-C
id<MTLCommandBuffer> commandBuffer = [_mCommandQueue commandBuffer];
```

In Metal, we need [`MTLComputeCommandEncoder`](https://developer.apple.com/documentation/metal/mtlcomputecommandencoder?language=objc) to write the command buffer.

```Objective-C
id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
```

### Set Function for Command Buffer

The main object for a command buffer is the "function", or we say `MTLComputePipelineState` in Metal.

```Objective-C
[computeEncoder setComputePipelineState:_mAddFunctionPSO];
```

### Set Arguments for Command Buffer

Apart from the "function", we also need to set arguments for the command buffer. Instead of using CUDA Runtime format, in Metal we need to set those with `MTLComputeCommandEncoder` methods [`setBuffer`](https://developer.apple.com/documentation/metal/mtlcomputecommandencoder/1443126-setbuffer?language=objc), [`setBytes`](https://developer.apple.com/documentation/metal/mtlcomputecommandencoder/1443159-setbytes?language=objc), etc.

## Final Push

### Finish Command Encoder

Once the function pipeline status and arguments are all set, we can end the encoding:

```Objective-C
[computeEncoder endEncoding];
```

### Commit Command Buffer

Once the encoder is finished, we can commit the command buffer, which means push the command buffer into the command queue:

```Objective-C
[commandBuffer commit];
```

### Wait until finished

The commit action of a command buffer is asynchronized, so before checking the data, we need to make sure the computation is finished:

```Objective-C
[commandBuffer waitUntilCompleted];
```

## Verify

Let's just see if the data computed in GPU is correct or not.