//
//  main.mm
//  quick-start-metal
//
//  Created by Wang Zhang on 5/29/20.
//  Copyright © 2020 Wang Zhang. All rights reserved.
//

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

int main(int argc, const char * argv[]) {
    @autoreleasepool {
        /* Context Setting */
        // Device
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            NSLog(@"No Metal devices found!");
            return 1;
        }
        // Command Queue
        id<MTLCommandQueue> commandQueue = [device newCommandQueue];
        if (commandQueue == nil) {
            NSLog(@"Failed to create command queue");
            return 1;
        }
                
        /* Data */
        uint32_t array_length = 128;
        size_t bytes = array_length * sizeof(float);
        
        // create buffer with `newBufferWithLength` and then set values
        id<MTLBuffer> bufferA = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        float* a_ptr = (float*)bufferA.contents;
        for (size_t i = 0; i<array_length; i++) {
            a_ptr[i] = (float)rand()/(float)(RAND_MAX);
        }
        
        // create array on host first and then use `newBufferWithBytes` to allocate and copy data
        float *b_ptr = (float *)malloc(bytes);
        for (size_t i = 0; i<array_length; i++) {
            b_ptr[i] = (float)rand()/(float)(RAND_MAX);
        }
        id<MTLBuffer> bufferB = [device newBufferWithBytes:b_ptr length:bytes options:MTLResourceStorageModeShared];
        
        // create buffer to hold result
        id<MTLBuffer> bufferC = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        
        /* Function */
        // Library
        id<MTLLibrary> defaultLibrary = [device newDefaultLibrary];
        if (defaultLibrary == nil) {
            NSLog(@"No valid metal files found!");
            return 1;
        }
        
        // Function
        id<MTLFunction> addVecFunc = [defaultLibrary newFunctionWithName:@"add_arrays"];
        if (addVecFunc == nil) {
            NSLog(@"Cannot find add_arrays in default library");
            return 1;
        }
        
        // Function Pipeline State
        NSError* error = nil;
        id<MTLComputePipelineState> addVecFuncPS = [device newComputePipelineStateWithFunction:addVecFunc error:&error];
        if (addVecFuncPS == nil) {
            NSLog(@"Failed to create pipeline state for add_array!");
            return 1;
        }
        
        /* Command Buffer */
        // Create Command Buffer
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        if (commandBuffer == nil) {
            NSLog(@"Failed to create command buffer");
            return 1;
        }
        
        // Create Encoder for the Command Buffer
        id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
        if (computeEncoder == nil) {
            NSLog(@"Failed to create compute command encoder");
            return 1;
        }
        
        // set compute pipeline state
        [computeEncoder setComputePipelineState:addVecFuncPS];
        
        // set arguments
        [computeEncoder setBuffer:bufferA offset:0 atIndex:0];
        [computeEncoder setBuffer:bufferB offset:0 atIndex:1];
        [computeEncoder setBuffer:bufferC offset:0 atIndex:2];
        
        // set thread and block
        // `dispatchThreadgroups:threadsPerThreadgroup:` is closer to the CUDA <<<block,thread>>> style
        // `dispatchThreads:threadsPerThreadgroup:` can help skip if/else diverge
        MTLSize gridSize = MTLSizeMake(array_length, 1, 1);
        NSUInteger threadGroupNumber = addVecFuncPS.maxTotalThreadsPerThreadgroup;
        if (threadGroupNumber > array_length) threadGroupNumber = array_length;
        MTLSize threadPerGroupSize  = MTLSizeMake(threadGroupNumber, 1, 1);
        [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadPerGroupSize];
        
        /* Final Push */
        // End the compute pass
        [computeEncoder endEncoding];
        
        // Execute the command
        [commandBuffer commit];
        
        // wait until finish
        [commandBuffer waitUntilCompleted];
        
        /* Verify */
        float* verify_ptr_a = (float*)bufferA.contents;
        float* verify_ptr_b = (float*)bufferB.contents;
        float* verify_ptr_c = (float*)bufferC.contents;
        
        for (size_t i=0; i<array_length; i++) {
            if (verify_ptr_a[i] + verify_ptr_b[i] != verify_ptr_c[i]) {
                NSLog(@"At position %zu: %f + %f != %f", i, verify_ptr_a[i], verify_ptr_b[i], verify_ptr_c[i]);
                return 1;
            } else {
                NSLog(@"At position %zu: %f + %f == %f", i, verify_ptr_a[i], verify_ptr_b[i], verify_ptr_c[i]);
            }
        }
        
        NSLog(@"Finished!");
    }
    return 0;
}
