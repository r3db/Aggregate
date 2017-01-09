using System;
using System.Runtime.InteropServices;
using Alea;
using Alea.CSharp;
using Alea.Parallel;

namespace Aggregate
{
    internal static class AggregateGpu
    {
        // GPU: Using Alea Parallel Linq!
        internal static T ComputeGpu0<T>(T[] array, Func<T, T, T> op)
        {
            return Gpu.Default.Aggregate(array, op);
        }
        
        // GPU: Interleaved Addressing! (Loop)
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
                gpu.Launch(() => KernelInterleavedAccess(arrayDevPtr, arrayLength, resultDevice, op), launchParams);

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

            const int maxThreads = 128;
            var threads = length < maxThreads ? np2(length) : maxThreads;
            var blocks = (length + threads - 1) / threads;
            var sharedMemory = threads <= 32 ? 2 * threads * Marshal.SizeOf<T>() : threads * Marshal.SizeOf<T>();

            //Console.WriteLine("Blocks : {0,7}, Threads: {1,7}, Shared-Memory: {2,7}, Length: {3,8}", blocks, threads, sharedMemory, length);

            return new LaunchParam(blocks, threads, sharedMemory);
        }

        private static void KernelInterleavedAccess<T>(deviceptr<T> array, int length, T[] result, Func<T, T, T> op)
        {
            var shared = __shared__.ExternArray<T>();

            var tid = threadIdx.x;
            var bid = blockIdx.x;
            var gid = blockDim.x * bid + tid;

            if (gid < length)
            {
                shared[tid] = array[gid];
            }

            DeviceFunction.SyncThreads();

            for (var s = 1; s < blockDim.x; s *= 2)
            {
                if (tid % (2 * s) == 0 && gid + s < length)
                {
                    shared[tid] = op(shared[tid], shared[tid + s]);
                }

                DeviceFunction.SyncThreads();
            }

            if (tid == 0)
            {
                result[bid] = shared[0];
            }
        }
    }
}