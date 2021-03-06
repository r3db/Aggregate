using System;
using System.Linq;
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
        private const int CpuThreashold = 1024;

        // Alea Parallel Linq!
        internal static T ComputeGpu0<T>(T[] array, Func<T, T, T> op)
        {
            return Gpu.Default.Aggregate(array, op);
        }
        
        // Interleaved Addressing!
        internal static T ComputeGpu1<T>(T[] array, Func<T, T, T> op)
        {
            return ReduceHelper(array, op, KernelInterleavedAccess, CreateLaunchParamsNonStridedAccess<T>);
        }

        // Sequential Addressing!
        internal static T ComputeGpu2<T>(T[] array, Func<T, T, T> op)
        {
            return ReduceHelper(array, op, KernelSequentialAccess, CreateLaunchParamsNonStridedAccess<T>);
        }

        // Sequential Reduce Idle Threads!
        internal static T ComputeGpu3<T>(T[] array, Func<T, T, T> op)
        {
            return ReduceHelper(array, op, KernelSequentialReduceIdleThreads, CreateLaunchParamsStridedAccess<T>);
        }

        // Sequential Warp!
        internal static T ComputeGpu4<T>(T[] array, Func<T, T, T> op)
        {
            return ReduceHelper(array, op, KernelSequentialReduceIdleThreadsWarp, CreateLaunchParamsStridedAccess<T>);
        }

        // Link: https://mail.google.com/mail/u/0/#inbox/1598d0b3b2850009?projector=1
        // I'm sure memory management is far from optimal!
        // Fixed Block and Thread!
        internal static T ComputeGpu5<T>(T[] array, Func<T, T, T> op)
        {
            const int dimGrid = 256;
            const int blockDim = 256;

            var gpu = Gpu.Default;

            var inputLength = array.Length;
            var inputMemory = gpu.ArrayGetMemory(array, true, false);
            var inputDevPtr = new deviceptr<T>(inputMemory.Handle);

            var resultMemory = gpu.AllocateDevice<T>(dimGrid);
            var resultDevPtr = new deviceptr<T>(resultMemory.Handle);

            gpu.Launch(() => KernelSequentialReduceIdleThreadsWarpMultiple(inputDevPtr, inputLength, resultDevPtr, op), new LaunchParam(dimGrid, blockDim));

            inputDevPtr = resultDevPtr;

            resultMemory = gpu.AllocateDevice<T>(dimGrid);
            resultDevPtr = new deviceptr<T>(resultMemory.Handle);

            gpu.Launch(() => KernelSequentialReduceIdleThreadsWarpMultiple(inputDevPtr, dimGrid, resultDevPtr, op), new LaunchParam(1, blockDim));

            return Gpu.CopyToHost(resultMemory)[0];
        }

        // I'm sure memory management is far from optimal!
        // Helpers
        private static T ReduceHelper<T>(T[] array, Func<T, T, T> op, Action<deviceptr<T>, int, T[], Func<T, T, T>> kernel, Func<int, LaunchParam> launchParamsFactory)
        {
            if (array.Length < CpuThreashold)
            {
                return array.AsParallel().Aggregate(op);
            }

            var gpu = Gpu.Default;

            var arrayLength = array.Length;
            var arrayMemory = gpu.ArrayGetMemory(array, true, false);
            var arrayDevPtr = new deviceptr<T>(arrayMemory.Handle);

            while (true)
            {
                var launchParams = launchParamsFactory(arrayLength);
                var resultLength = launchParams.GridDim.x;
                var resultDevice = gpu.Allocate<T>(resultLength);

                // I'm allowed to use the CPU.
                if (arrayLength < CpuThreashold)
                {
                    using (var m = arrayMemory as Memory<T>)
                    {
                        return Gpu.CopyToHost(m).AsParallel().Aggregate(op);
                    }
                }

                // ReSharper disable once AccessToModifiedClosure
                // ReSharper disable once AccessToModifiedClosure
                gpu.Launch(() => kernel(arrayDevPtr, arrayLength, resultDevice, op), launchParams);

                // I should be able to dispose at this point!
                // This is a symptom I did something stupid!
                //arrayMemory.Dispose();

                arrayLength = resultLength;
                arrayMemory = gpu.ArrayGetMemory(resultDevice, true, false);
                arrayDevPtr = new deviceptr<T>(arrayMemory.Handle);
            }
        }

        private static LaunchParam CreateLaunchParamsNonStridedAccess<T>(int length)
        {
            var threads = length < MaxThreads
                ? NextPowerOfTwo(length)
                : MaxThreads;

            var blocks = (length + threads - 1) / threads;
            var sharedMemory = threads * Marshal.SizeOf<T>();

            PrintLaunchParamInformation(length, blocks, threads, sharedMemory);
            return new LaunchParam(blocks, threads, sharedMemory);
        }

        private static LaunchParam CreateLaunchParamsStridedAccess<T>(int length)
        {
            var threads = length < 2 * MaxThreads 
                ? NextPowerOfTwo((length + 1) / 2)
                : MaxThreads;

            var blocks = (length + 2 * threads - 1) / (2 * threads);
            var sharedMemory = threads * Marshal.SizeOf<T>();

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

            shared[tid] = (gid < length && gid + blockDim.x < length)
                ? op(array[gid], array[gid + blockDim.x])
                : array[gid];

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

            shared[tid] = (gid < length && gid + bdm < length)
                ? op(array[gid], array[gid + bdm])
                : array[gid];

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

        // ReSharper disable once SuggestBaseTypeForParameter
        private static void KernelSequentialReduceIdleThreadsWarpMultiple<T>(deviceptr<T> array, int length, deviceptr<T> result, Func<T, T, T> op)
        {
            var tid = threadIdx.x;
            var bid = blockIdx.x;
            var bdm = blockDim.x;
            var gid = bdm * bid + tid;

            // Todo: 'default(T)' is a bad idea, think of (n * 0) => The accumulator's initial value should be provided by the user!
            var accumulator = default(T);

            while (gid < length)
            {
                accumulator = op(accumulator, array[gid]);
                gid += gridDim.x * bdm;
            }

            accumulator = op(accumulator, DeviceFunction.ShuffleDown(accumulator, 16));
            accumulator = op(accumulator, DeviceFunction.ShuffleDown(accumulator,  8));
            accumulator = op(accumulator, DeviceFunction.ShuffleDown(accumulator,  4));
            accumulator = op(accumulator, DeviceFunction.ShuffleDown(accumulator,  2));
            accumulator = op(accumulator, DeviceFunction.ShuffleDown(accumulator,  1));

            var shared = __shared__.Array<T>(8);

            if (tid % WarpSize == 0)
            {
                shared[tid / WarpSize] = accumulator;
            }

            DeviceFunction.SyncThreads();
            
            if (tid == 0)
            {
                var a = op(op(shared[0], shared[1]), op(shared[2], shared[3]));
                var b = op(op(shared[4], shared[5]), op(shared[6], shared[7]));
                result[bid] = op(a, b);
            }
        }
    }
}