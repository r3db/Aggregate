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
    // http://docs.nvidia.com/gameworks/index.html#developertools/desktop/debugging_cuda_application.htm
    // http://http.developer.nvidia.com/NsightVisualStudio/3.0/Documentation/UserGuide/HTML/Content/CUDA_Warp_Watch.htm
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

        // GPU: Interleaved Addressing!
        [GpuManaged]
        internal static T ComputeGpu2<T>(T[] array, Func<T, T, T> op)
        {
            var lp = CreateLaunchParam<T>(array);
            var resultSize = lp.GridDim.x;
            var result = new T[resultSize];

            Gpu.Default.Launch(() =>
            {
                var shared = __shared__.ExternArray<T>();
                
                var tid = threadIdx.x;
                var bid = blockIdx.x;
                var gid = blockDim.x * bid + tid;

                if (tid >= array.Length)
                {
                    return;
                }

                shared[tid] = array[gid];
                DeviceFunction.SyncThreads();

                for (var s = 1; s < blockDim.x; s *= 2)
                {
                    if (tid % (2 * s) == 0)
                    {
                        shared[tid] = op(shared[tid], shared[tid + s]);
                    }

                    DeviceFunction.SyncThreads();
                }

                if (tid == 0)
                {
                    result[bid] = shared[0];
                }
            }, lp);

            Thread.Sleep(50);
            return resultSize > 1 ? ComputeGpu2(result, op) : result[0];
        }

        private static LaunchParam CreateLaunchParam<T>(T[] array)
        {
            var attributes = Gpu.Default.Device.Attributes;

            var maxThreads = attributes.MaxThreadsPerBlock;
            var threads = array.Length < maxThreads ? NextPowerOfTwo(array.Length) : maxThreads;
            var blocks = (array.Length + threads - 1) / threads;
            var sharedMemory = threads <= 32 ? 2 * threads * Marshal.SizeOf<T>() : threads * Marshal.SizeOf<T>();

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
    }
}