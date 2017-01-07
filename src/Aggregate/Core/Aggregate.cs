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
        // Done!
        // CPU: Using Sequential Loop!
        internal static int ComputeCpu1(int[] array)
        {
            var result = 0;

            // ReSharper disable once ForCanBeConvertedToForeach
            // ReSharper disable once LoopCanBeConvertedToQuery
            for (var i = 0; i < array.Length; i++)
            {
                result += array[i];
            }

            return result;
        }

        // Done!
        // CPU: Using Parallel ForEach!
        internal static int ComputeCpu2(int[] array)
        {
            var result = 0;

            Parallel.ForEach(array, () => 0, (value, state, local) => local + value, x =>
            {
                Interlocked.Add(ref result, x);
            });

            return result;
        }

        // CPU: Using Linq!
        internal static int ComputeCpu3(int[] array)
        {
            return array.Aggregate(0, (a, b) => a + b);
        }

        // Done!
        // CPU: Using Parallel Linq!
        internal static int ComputeCpu4(int[] array)
        {
            return array.AsParallel().Aggregate(0, (a, b) => a + b);
        }

        // Done!
        // GPU: Using Alea Aggregate!
        internal static int ComputeGpu1(int[] array)
        {
            return Gpu.Default.Aggregate(array, (a, b) => a + b);
        }

        // GPU: Interleaved Addressing!
        internal static int ComputeGpu2(int[] array)
        {
            var tb = 1024;
            var bc = (array.Length + (tb - 1)) / tb;
            var lp = CreateLaunchParam(bc, tb, tb * sizeof(int));
            var result = new int[bc];

            Gpu.Default.Launch(() =>
            {
                var shared = __shared__.ExternArray<int>();
                
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
                        shared[tid] += shared[tid + s];
                    }

                    DeviceFunction.SyncThreads();
                }

                if (tid == 0)
                {
                    result[bid] = shared[0];
                }
            }, lp);

            return bc > 1 ? ComputeGpu2(result) : result[0];
        }

        private static LaunchParam CreateLaunchParam(int gridDim, int blockDim, int sharedMemorySize)
        {
            //Console.WriteLine("-------------");
            //Console.WriteLine("MaxBlocks : {0}, Current Block Allocation : {1}", ushort.MaxValue, gridDim);
            //Console.WriteLine("MaxThreads: {0}, Current Thread Allocation: {1}", Gpu.Default.Device.Attributes.MaxThreadsPerBlock, blockDim);
            //Console.WriteLine("MaxShared : {0}, Current Shared Allocation: {1}", Gpu.Default.Device.Attributes.MaxSharedMemoryPerBlock, sizeof(int) * blockDim);

            return new LaunchParam(gridDim, blockDim, sharedMemorySize);
        }

        //internal static int ComputeGpu2(int[] array)
        //{
        //    var threads = 1024;
        //    var blocks = array.Length / threads;

        //    var result = new int[threads];
        //    var lp = new LaunchParam(blocks, threads, threads * sizeof(int));

        //    Gpu.Default.Launch(() =>
        //    {
        //        var shared = __shared__.ExternArray<int>();
        //        var myId = threadIdx.x + blockDim.x * blockIdx.x;
        //        var tid  = threadIdx.x;

        //        shared[tid] = array[myId];
        //        DeviceFunction.SyncThreads();

        //        for (int s = blockDim.x / 2; s > 0; s >>= 1)
        //        {
        //            if (tid < s)
        //            {
        //                shared[tid] += shared[tid + s];
        //            }

        //            DeviceFunction.SyncThreads();
        //        }

        //        if (tid == 0)
        //        {
        //            result[blockIdx.x] = shared[0];
        //        }
        //    }, lp);

        //    // Todo: This aggregate is not yet fully functional!
        //    //if (result.Length > 1)
        //    //{
        //    //    return ComputeGpu2(result);
        //    //}

        //    return result.Sum();
        //}
    }
}