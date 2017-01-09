using System;
using System.Runtime.InteropServices;
using Alea;
using Alea.CSharp;
using Alea.Parallel;

namespace Aggregate
{
    internal static class AggregateGpu
    {
        private const int WarpSize = 32;
        private const int MaxThreads = 128;

        // GPU: Using Alea Parallel Linq!
        internal static T ComputeGpu0<T>(T[] array, Func<T, T, T> op)
        {
            return Gpu.Default.Aggregate(array, op);
        }
        
        // GPU: Interleaved Addressing!
        internal static T ComputeGpu1<T>(T[] array, Func<T, T, T> op)
        {
            var gpu = Gpu.Default;

            var arrayLength = array.Length;
            var arrayMemory = gpu.ArrayGetMemory(array, true, false);
            var arrayDevPtr = new deviceptr<T>(arrayMemory.Handle);

            while (true)
            {
                var launchParams = CreateLaunchParamsNonStridedAccess<T>(arrayLength);
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

        // GPU: Sequential Addressing!
        internal static T ComputeGpu2<T>(T[] array, Func<T, T, T> op)
        {
            var gpu = Gpu.Default;

            var arrayLength = array.Length;
            var arrayMemory = gpu.ArrayGetMemory(array, true, false);
            var arrayDevPtr = new deviceptr<T>(arrayMemory.Handle);

            while (true)
            {
                var launchParams = CreateLaunchParamsNonStridedAccess<T>(arrayLength);
                var resultLength = launchParams.GridDim.x;
                var resultDevice = gpu.Allocate<T>(resultLength);

                // ReSharper disable once AccessToModifiedClosure
                // ReSharper disable once AccessToModifiedClosure
                gpu.Launch(() => KernelSequentialAccess(arrayDevPtr, arrayLength, resultDevice, op), launchParams);

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

        // GPU: Sequential Reduce Idle Threads!
        internal static T ComputeGpu3<T>(T[] array, Func<T, T, T> op)
        {
            var gpu = Gpu.Default;

            var arrayLength = array.Length;
            var arrayMemory = gpu.ArrayGetMemory(array, true, false);
            var arrayDevPtr = new deviceptr<T>(arrayMemory.Handle);

            while (true)
            {
                var launchParams = CreateLaunchParamsStridedAccess<T>(arrayLength);
                var resultLength = launchParams.GridDim.x;
                var resultDevice = gpu.Allocate<T>(resultLength);

                // ReSharper disable once AccessToModifiedClosure
                // ReSharper disable once AccessToModifiedClosure
                gpu.Launch(() => KernelSequentialReduceIdleThreads(arrayDevPtr, arrayLength, resultDevice, op), launchParams);

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

        // GPU: Sequential Warp!
        internal static T ComputeGpu4<T>(T[] array, Func<T, T, T> op)
        {
            var gpu = Gpu.Default;

            var arrayLength = array.Length;
            var arrayMemory = gpu.ArrayGetMemory(array, true, false);
            var arrayDevPtr = new deviceptr<T>(arrayMemory.Handle);

            while (true)
            {
                var launchParams = CreateLaunchParamsStridedAccess<T>(arrayLength);
                var resultLength = launchParams.GridDim.x;
                var resultDevice = gpu.Allocate<T>(resultLength);

                // ReSharper disable once AccessToModifiedClosure
                // ReSharper disable once AccessToModifiedClosure
                gpu.Launch(() => KernelSequentialReduceIdleThreadsWarp(arrayDevPtr, arrayLength, resultDevice, op), launchParams);

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
        private static LaunchParam CreateLaunchParamsNonStridedAccess<T>(int length)
        {
            var threads = length < MaxThreads ? NextPowerOfTwo(length) : MaxThreads;
            var blocks = (length + threads - 1) / threads;
            var sharedMemory = threads <= WarpSize ? 2 * threads * Marshal.SizeOf<T>() : threads * Marshal.SizeOf<T>();

            PrintLaunchParamInformation(length, blocks, threads, sharedMemory);
            return new LaunchParam(blocks, threads, sharedMemory);
        }

        private static LaunchParam CreateLaunchParamsStridedAccess<T>(int length)
        {
            const int maxThreads = 128;
            var threads = length < 2 * maxThreads ? NextPowerOfTwo((length + 1) / 2) : maxThreads;
            var blocks = (length + (2 * threads) - 1) / (2 * threads);
            var sharedMemory = threads <= WarpSize ? 2 * threads * Marshal.SizeOf<T>() : threads * Marshal.SizeOf<T>();

            PrintLaunchParamInformation(length, blocks, threads, sharedMemory);
            return new LaunchParam(blocks, threads, sharedMemory);
        }
        
        private static int NextPowerOfTwo(int value) {
            --value;
            value |= value >> 1;
            value |= value >> 2;
            value |= value >> 4;
            value |= value >> 8;
            value |= value >> 16;

            return ++value;
        }

        private static void PrintLaunchParamInformation(int length, int blocks, int threads, int sharedMemory)
        {
            //Console.WriteLine("Blocks : {0,7}, Threads: {1,7}, Shared-Memory: {2,7}, Length: {3,8}", blocks, threads, sharedMemory, length);
        }

        // ReSharper disable once SuggestBaseTypeForParameter
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

        // ReSharper disable once SuggestBaseTypeForParameter
        private static void KernelSequentialAccess<T>(deviceptr<T> array, int length, T[] result, Func<T, T, T> op)
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

            for (var s = blockDim.x / 2; s > 0; s >>= 1)
            {
                if (tid < s && gid + s < length)
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

        // ReSharper disable once SuggestBaseTypeForParameter
        private static void KernelSequentialReduceIdleThreads<T>(deviceptr<T> array, int length, T[] result, Func<T, T, T> op)
        {
            var shared = __shared__.ExternArray<T>();

            var tid = threadIdx.x;
            var bid = blockIdx.x;
            var gid = 2 * blockDim.x * bid + tid;

            shared[tid] = (gid < length && gid + blockDim.x < length) ? op(array[gid], array[gid + blockDim.x]) : array[gid];

            DeviceFunction.SyncThreads();

            for (int s = blockDim.x / 2; s > 0; s >>= 1)
            {
                if (tid < s && gid + s < length)
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

        // ReSharper disable once SuggestBaseTypeForParameter
        private static void KernelSequentialReduceIdleThreadsWarp<T>(deviceptr<T> array, int length, T[] result, Func<T, T, T> op)
        {
            var shared = __shared__.ExternArray<T>();

            var tid = threadIdx.x;
            var bid = blockIdx.x;
            var bdm = blockDim.x;
            var gid = 2 * bdm * bid + tid;

            shared[tid] = (gid < length && gid + bdm < length) ? op(array[gid], array[gid + bdm]) : array[gid];

            DeviceFunction.SyncThreads();

            for (var s = bdm / 2; s > WarpSize; s >>= 1)
            {
                if (tid < s && gid + s < length)
                {
                    shared[tid] = op(shared[tid], shared[tid + s]);
                }

                DeviceFunction.SyncThreads();
            }

            if (tid < WarpSize)
            {
                if (bdm >= 2 * WarpSize)
                {
                    shared[tid] = op(shared[tid], shared[tid + WarpSize]);
                }

                shared[tid] = op(shared[tid], DeviceFunction.ShuffleDown(shared[tid], 16));
                shared[tid] = op(shared[tid], DeviceFunction.ShuffleDown(shared[tid], 8));
                shared[tid] = op(shared[tid], DeviceFunction.ShuffleDown(shared[tid], 4));
                shared[tid] = op(shared[tid], DeviceFunction.ShuffleDown(shared[tid], 2));
                shared[tid] = op(shared[tid], DeviceFunction.ShuffleDown(shared[tid], 1));
            }

            if (tid == 0)
            {
                result[bid] = shared[0];
            }
        }
    }
}