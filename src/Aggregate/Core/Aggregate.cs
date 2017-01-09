using System;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using Alea;
using Alea.CSharp;
using Alea.Parallel;
using System.Threading;

namespace Aggregate
{
    internal static class Aggregate
    {
        // Done!
        // CPU: Using Sequential Loop!
        internal static T ComputeCpu1<T>(T[] array, Func<T, T, T> op)
        {
            var result = default(T);

            // ReSharper disable once ForCanBeConvertedToForeach
            // ReSharper disable once LoopCanBeConvertedToQuery
            for (var i = 0; i < array.Length; i++)
            {
                result = op(result, array[i]);
            }

            return result;
        }

        // CPU: Using Linq!
        // ReSharper disable once ParameterTypeCanBeEnumerable.Global
        internal static T ComputeCpu2<T>(T[] array, Func<T, T, T> op)
        {
            return array.Aggregate(op);
        }

        // Done!
        // CPU: Using Parallel ForEach!
        // ReSharper disable once ParameterTypeCanBeEnumerable.Global
        internal static T ComputeCpu3<T>(T[] array, Func<T, T, T> op)
        {
            var result = default(T);
            var locker = new object();

            Parallel.ForEach(array, () => default(T), (value, state, local) => op(local, value), x =>
            {
                // Todo: There are better ways of doing this!
                lock (locker)
                {
                    result = op(result, x);
                }
            });

            return result;
        }

        // Done!
        // CPU: Using Parallel Linq!
        // ReSharper disable once ParameterTypeCanBeEnumerable.Global
        internal static T ComputeCpu4<T>(T[] array, Func<T, T, T> op)
        {
            return array.AsParallel().Aggregate(op);
        }

        // Done!
        // GPU: Using Alea Aggregate!
        internal static T ComputeGpu1<T>(T[] array, Func<T, T, T> op)
        {
            return Gpu.Default.Aggregate(array, op);
        }




        // GPU: Interleaved Addressing! (Recursive)
        [GpuManaged]
        internal static T ComputeGpu2<T>(T[] array, Func<T, T, T> op)
        {
            var lp = CreateLaunchParam<T>(array.Length);
            var resultSize = lp.GridDim.x;
            var result = new T[resultSize];

            Gpu.Default.Launch(() => {
                InterleavedAddressingKernel(array, result, op);
            }, lp);

            return resultSize > 1 ? ComputeGpu2(result, op) : result[0];
        }

        // GPU: Interleaved Addressing! (Loop)
        internal static T ComputeGpu3<T>(T[] array, Func<T, T, T> op)
        {
            var inputDevice  = Gpu.Default.Allocate<T>(array);

            while (true)
            {
                var lp = CreateLaunchParam<T>(Gpu.ArrayGetLength(inputDevice));
                var resultDevice = Gpu.Default.Allocate<T>(lp.GridDim.x);

                //Console.WriteLine(Gpu.ArrayGetLength(inputDevice));
                //Console.WriteLine(Gpu.ArrayGetLength(resultDevice));

                Gpu.Default.Launch(() => {
                    InterleavedAddressingKernel(inputDevice, resultDevice, op);
                }, lp);

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

        // GPU: Interleaved Addressing! (Method)
        internal static T ComputeGpu4<T>(T[] array, Func<T, T, T> op)
        {
            while (true)
            {
                var result = ComputeGpu4Internal(array, op);

                if (result.Length == 1)
                {
                    return result[0];
                }

                array = result;
            }
        }

        // GPU: Interleaved Addressing! (Loop)
        private static T[] ComputeGpu4Internal<T>(T[] array, Func<T, T, T> op)
        {
            var lp = CreateLaunchParam<T>(array.Length);
            var resultDevice = Gpu.Default.Allocate<T>(lp.GridDim.x);

            Gpu.Default.Launch(() => {
                InterleavedAddressingKernel(array, resultDevice, op);
            }, lp);

            return Gpu.CopyToHost(resultDevice);
        }





        // GPU: Sequential Addressing! (Recursive)
        [GpuManaged]
        internal static T ComputeGpu5<T>(T[] array, Func<T, T, T> op)
        {
            var lp = CreateLaunchParam<T>(array.Length);
            var resultSize = lp.GridDim.x;
            var result = new T[resultSize];

            Gpu.Default.Launch(() => {
                SequentialAddressingKernel(array, result, op);
            }, lp);

            return resultSize > 1 ? ComputeGpu2(result, op) : result[0];
        }

        // GPU: Sequential Addressing! (Loop)
        internal static T ComputeGpu6<T>(T[] array, Func<T, T, T> op)
        {
            var inputDevice = Gpu.Default.Allocate<T>(array);

            while (true)
            {
                var lp = CreateLaunchParam<T>(Gpu.ArrayGetLength(inputDevice));
                var resultDevice = Gpu.Default.Allocate<T>(lp.GridDim.x);

                //Console.WriteLine(Gpu.ArrayGetLength(inputDevice));
                //Console.WriteLine(Gpu.ArrayGetLength(resultDevice));

                Gpu.Default.Launch(() => {
                    SequentialAddressingKernel(inputDevice, resultDevice, op);
                }, lp);

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

        // GPU: Sequential Addressing! (Method)
        internal static T ComputeGpu7<T>(T[] array, Func<T, T, T> op)
        {
            while (true)
            {
                var result = ComputeGpu7Internal(array, op);

                if (result.Length == 1)
                {
                    return result[0];
                }

                array = result;
            }
        }

        private static T[] ComputeGpu7Internal<T>(T[] array, Func<T, T, T> op)
        {
            var lp = CreateLaunchParam<T>(array.Length);
            var resultDevice = Gpu.Default.Allocate<T>(lp.GridDim.x);

            Gpu.Default.Launch(() => {
                SequentialAddressingKernel(array, resultDevice, op);
            }, lp);

            return Gpu.CopyToHost(resultDevice);
        }




        private static LaunchParam CreateLaunchParam<T>(int length)
        {
            var attributes = Gpu.Default.Device.Attributes;

            var maxThreads = attributes.MaxThreadsPerBlock;
            var threads = length < maxThreads ? NextPowerOfTwo(length) : maxThreads;
            var blocks = (length + threads - 1) / threads;
            var sharedMemory = threads <= 32 ? 2 * threads * Marshal.SizeOf<T>() : threads * Marshal.SizeOf<T>();

            //Console.WriteLine("Blocks : {0,5}, Threads: {1,5}, Shared-Memory: {2,5}", blocks, threads, sharedMemory);

            return new LaunchParam(blocks, threads, sharedMemory);
        }

        private static int NextPowerOfTwo(int n)
        {
            --n;
            n |= n >> 1;
            n |= n >> 2;
            n |= n >> 4;
            n |= n >> 8;
            n |= n >> 16;
            return ++n;
        }

        // Kernels
        private static void InterleavedAddressingKernel<T>(T[] array, T[] resultDevice, Func<T, T, T> op)
        {
            var shared = __shared__.ExternArray<T>();

            var tid = threadIdx.x;
            var bid = blockIdx.x;
            var gid = blockDim.x * bid + tid;

            if (gid < array.Length)
            {
                shared[tid] = array[gid];
            }

            DeviceFunction.SyncThreads();

            for (var s = 1; s < blockDim.x; s *= 2)
            {
                if (tid % (2 * s) == 0 && gid + s < array.Length)
                {
                    shared[tid] = op(shared[tid], shared[tid + s]);
                }

                DeviceFunction.SyncThreads();
            }

            if (tid == 0)
            {
                resultDevice[bid] = shared[0];
            }
        }

        private static void SequentialAddressingKernel<T>(T[] array, T[] resultDevice, Func<T, T, T> op)
        {
            var shared = __shared__.ExternArray<T>();

            var tid = threadIdx.x;
            var bid = blockIdx.x;
            var gid = blockDim.x * bid + tid;

            if (gid < array.Length)
            {
                shared[tid] = array[gid];
            }

            DeviceFunction.SyncThreads();

            for (int s = blockDim.x / 2; s > 0; s >>= 1)
            {
                if (tid < s && gid + s < array.Length)
                {
                    shared[tid] = op(shared[tid], shared[tid + s]);
                }

                DeviceFunction.SyncThreads();
            }

            if (tid == 0)
            {
                resultDevice[bid] = shared[0];
            }
        }
    }
}