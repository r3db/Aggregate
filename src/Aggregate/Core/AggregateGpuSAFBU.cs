﻿using System;
using System.Runtime.InteropServices;
using Alea;
using Alea.CSharp;

namespace Aggregate
{
    internal static class AggregateGpuSAFBU
    {
        // GPU: Sequential Addressing Fully Busy! (Loop)
        internal static T ComputeGpu1<T>(T[] array, Func<T, T, T> op)
        {
            var gpu = Gpu.Default;

            var arrayLength = array.Length;
            var arrayMemory = gpu.ArrayGetMemory(array, true, false);
            var arrayDevPtr = new deviceptr<T>(arrayMemory.Handle);

            while (true)
            {
                var launchParams = CreateLaunchParams<T>(arrayLength);
                var resultLength = launchParams.GridDim.x;
                var resultDevice = gpu.Allocate<T>(resultLength);

                // ReSharper disable once AccessToModifiedClosure
                // ReSharper disable once AccessToModifiedClosure
                gpu.Launch(() => Kernel(arrayDevPtr, arrayLength, resultDevice, op), launchParams);

                if (resultLength == 1)
                {
                    arrayMemory.Dispose();
                    var result = Gpu.CopyToHost(resultDevice);
                    Gpu.Free(resultDevice);
                    return result[0];
                }

                // I should be able to dispose at this point!
                // This is a symptom I did something stupid!
                //arrayMemory.Dispose();

                arrayLength = resultLength;
                arrayMemory = gpu.ArrayGetMemory(resultDevice, true, false);
                arrayDevPtr = new deviceptr<T>(arrayMemory.Handle);
            }
        }

        // Helpers
        private static LaunchParam CreateLaunchParams<T>(int length)
        {
            Func<int, int> np2 = n => {
                --n;
                n |= n >> 1;
                n |= n >> 2;
                n |= n >> 4;
                n |= n >> 8;
                n |= n >> 16;

                return ++n;
            };

            var attributes = Gpu.Default.Device.Attributes;

            var maxThreads = 128;
            var threads = length < 2 * maxThreads ? np2((length + 1) / 2) : maxThreads;
            var blocks = (length + (2 * threads) - 1) / (2 * threads);
            var sharedMemory = threads <= 32 ? 2 * threads * Marshal.SizeOf<T>() : threads * Marshal.SizeOf<T>();

            //Console.WriteLine("Blocks : {0,7}, Threads: {1,7}, Shared-Memory: {2,7}, Length: {3,8}", blocks, threads, sharedMemory, length);

            return new LaunchParam(blocks, threads, sharedMemory);
        }

        private static void Kernel<T>(deviceptr<T> array, int length, T[] resultDevice, Func<T, T, T> op)
        {
            var shared = __shared__.ExternArray<T>();

            var tid = threadIdx.x;
            var bid = blockIdx.x;
            var bdm = blockDim.x;
            var gid = 2 * bdm * bid + tid;

            shared[tid] = (gid < length && gid + bdm < length) ? op(array[gid], array[gid + bdm]) : array[gid];

            DeviceFunction.SyncThreads();

            for (var s = bdm / 2; s > 32; s >>= 1)
            {
                if (tid < s && gid + s < length)
                {
                    shared[tid] = op(shared[tid], shared[tid + s]);
                }

                DeviceFunction.SyncThreads();
            }

            if (tid < 32)
            {
                if (bdm >= 64)
                {
                    shared[tid] = op(shared[tid], shared[tid + 32]);
                }

                for (var offset = 32 / 2; offset > 0; offset /= 2)
                {
                    shared[tid] = op(shared[tid], DeviceFunction.ShuffleDown(shared[tid], offset));
                }
            }

            if (tid == 0)
            {
                resultDevice[bid] = shared[0];
            }
        }
    }
}