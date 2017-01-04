using System;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Alea;
using Alea.CSharp;
using Alea.Parallel;

namespace Aggregate
{
    // http://docs.nvidia.com/gameworks/index.html#developertools/desktop/debugging_cuda_application.htm
    // http://http.developer.nvidia.com/NsightVisualStudio/3.0/Documentation/UserGuide/HTML/Content/CUDA_Warp_Watch.htm
    internal static class Aggregate
    {
        // CPU: Using Sequential Loop!
        internal static long ComputeCpu1(int[] array)
        {
            var result = 0L;

            // ReSharper disable once LoopCanBeConvertedToQuery
            for (int i = 0; i < array.Length; i++)
            {
                result += array[i];
            }

            return result;
        }

        // CPU: Using Parallel ForEach!
        internal static long ComputeCpu2(int[] array)
        {
            var result = 0L;

            Parallel.ForEach(array, () => 0L, (value, state, local) => local + value, x =>
            {
                Interlocked.Add(ref result, x);
            });

            return result;
        }

        // CPU: Using Linq!
        internal static long ComputeCpu3(int[] array)
        {
            return array.Aggregate(0L, (a, b) => a + b);
        }

        // CPU: Using Parallel Linq!
        internal static long ComputeCpu4(int[] array)
        {
            return array.AsParallel().Aggregate(0L, (a, b) => a + b);
        }

        // GPU: Using Aggregate!
        internal static long ComputeGpu1(int[] array)
        {
            // Todo: What I really wanted was to return Int64.
            return Gpu.Default.Aggregate(array, (a, b) => a + b);
        }

        internal static long ComputeGpu2(int[] array)
        {
            var threads = 1024;
            var blocks = array.Length / threads;

            var result = new long[threads];
            var lp = new LaunchParam(blocks, threads, threads * sizeof(int));

            Gpu.Default.Launch(() =>
            {
                var shared = __shared__.ExternArray<int>();
                var myId = threadIdx.x + blockDim.x * blockIdx.x;
                var tid  = threadIdx.x;

                shared[tid] = array[myId];
                DeviceFunction.SyncThreads();

                for (int s = blockDim.x / 2; s > 0; s >>= 1)
                {
                    if (tid < s)
                    {
                        shared[tid] += shared[tid + s];
                    }

                    DeviceFunction.SyncThreads();
                }

                if (tid == 0)
                {
                    result[blockIdx.x] = shared[0];
                }
            }, lp);

            // Todo: This aggregate is not yet fully functional!
            //if (result.Length > 1)
            //{
            //    return ComputeGpu2(result);
            //}

            return result.Sum();
        }
    }
}