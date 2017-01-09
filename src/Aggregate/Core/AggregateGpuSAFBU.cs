﻿using System;
using System.Runtime.InteropServices;
using Alea;
using Alea.CSharp;
using System.Diagnostics;

namespace Aggregate
{
    internal static class AggregateGpuSAFBU
    {
        // GPU: Sequential Addressing Fully Busy! (Recursive)
        [GpuManaged]
        internal static T ComputeGpu1<T>(T[] array, Func<T, T, T> op)
        {
            var lp = CreateLaunchParam<T>(array.Length);
            var resultSize = lp.GridDim.x;
            var result = new T[resultSize];

            Gpu.Default.Launch(() => Kernel(array, result, op), lp);

            return resultSize > 1 ? ComputeGpu1(result, op) : result[0];
        }

        // GPU: Sequential Addressing Fully Busy! (Loop)
        internal static T ComputeGpu2<T>(T[] array, Func<T, T, T> op)
        {
            var inputDevice = Gpu.Default.Allocate<T>(array);

            while (true)
            {
                var lp = CreateLaunchParam<T>(Gpu.ArrayGetLength(inputDevice));
                var resultDevice = Gpu.Default.Allocate<T>(lp.GridDim.x);

                Gpu.Default.Launch(() => Kernel(inputDevice, resultDevice, op), lp);

                if (Gpu.ArrayGetLength(resultDevice) == 1)
                {
                    var result = Gpu.CopyToHost<T>(resultDevice);

                    Gpu.Free(inputDevice);
                    Gpu.Free(resultDevice);

                    return result[0];
                }

                Gpu.Free(inputDevice);
                inputDevice = resultDevice;
            }
        }

        // Helpers
        private static LaunchParam CreateLaunchParam<T>(int length)
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

            var maxThreads = attributes.MaxThreadsPerBlock;
            var threads = length < 2 * maxThreads ? np2((length + 1) / 2) : maxThreads;
            var blocks = (length + (2 * threads) - 1) / (2 * threads);
            var sharedMemory = threads <= 32 ? 2 * threads * Marshal.SizeOf<T>() : threads * Marshal.SizeOf<T>();

            //Console.WriteLine("Blocks : {0,7}, Threads: {1,7}, Shared-Memory: {2,7}, Length: {3,8}", blocks, threads, sharedMemory, length);

            return new LaunchParam(blocks, threads, sharedMemory);
        }

        private static void Kernel<T>(T[] array, T[] resultDevice, Func<T, T, T> op)
        {
            var shared = __shared__.ExternArray<T>();

            var tid = threadIdx.x;
            var bid = blockIdx.x;
            var gid = 2 * blockDim.x * bid + tid;

            shared[tid] = (gid < array.Length && gid + blockDim.x < array.Length)
                ? op(array[gid], array[gid + blockDim.x])
                : array[gid];

            DeviceFunction.SyncThreads();

            for (int s = blockDim.x / 2; s > 32; s >>= 1)
            {
                if (tid < s && gid + s < array.Length)
                {
                    shared[tid] = op(shared[tid], shared[tid + s]);
                }

                DeviceFunction.SyncThreads();
            }

            if (tid < 32)
            {
                // Fetch final intermediate sum from 2nd warp
                if (blockDim.x >= 64)
                {
                    shared[tid] = op(shared[tid], shared[tid + 32]);
                }

                // Reduce final warp using shuffle
                for (int offset = 32 / 2; offset > 0; offset /= 2)
                {
                    shared[tid] = op(shared[tid], DeviceFunction.ShuffleDown<T>(shared[tid], offset));
                }
            }

            if (tid == 0)
            {
                resultDevice[bid] = shared[0];
            }
        }
    }
}