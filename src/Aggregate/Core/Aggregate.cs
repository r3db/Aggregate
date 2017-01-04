using System;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Alea;
using Alea.Parallel;

namespace Aggregate
{
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
    }
}