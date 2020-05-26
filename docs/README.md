# Documentation for Metal and Metal Performance Shaders on Machine Learning

Here is an unofficial documentation for Metal and MPS (Metal Performance Shaders) from Apple. The assumed readers are ML (machine learning) engineers as only topics about how to use Metal and MPS on ML will be covered. Readers are supposed to have some very basic understanding of the following things:

- machine learning concept *(like functionality of ops)*
- C++ *(Objective-C? even better)*
- CUDA *(OpenCL? even better)*

Both Metal and MPS offers two kinds of bindings: Swift and Objective-C. For a better interpolation between C++ and Python, which are essentially used in machine learning, here we choose Objective-C binding.

If you are not familiar with Objective-C, here is a [quick introduction](./quick-objectivec.md) for this language.

## Metal

Metal is the fundamental library providing GPU computation features to Apple developers. If you are familiar with OpenCL, Metal is just like a combination of OpenCL and OpenGL on Apple platform. If you never uses OpenCL but know CUDA for a while, then Metal is like the CUDA **Driver** API. It introduces basic concepts, such like device, command queue, command buffer, etc. This [quick start](./example-vecadd.md) shows how to add two vectors on GPU with Metal.

## MPS (Metal Performance Shaders)

Writing all kinds of kernel for neural network operations, like convolution, matrix multiplication, element-wise activation, can be exhausting. Luckily, Apple released MPS with a bunch of pre-configured kernels on neural network. You may compare it to cuDNN. This [quick start](./example-fc.md) is about showing all the concepts in neural network with Metal and MPS rather than a complete network network performing some tasks. 

## Train a CNN

Here use an [example](./example-cnn-train.md) from Apple to show how to train a convolution neural network with Metal and MPS.

## Train a RNN

Finally, let's create a recurrent neural network and see [how to train](./example-rnn-train.md) it.