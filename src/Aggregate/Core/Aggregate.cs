using System;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace Aggregate
{
    internal static class Aggregate
    {
        // CPU: (Sum) Using Sequential Loop!
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

        // CPU: (Sum) Using Parallel ForEach!
        internal static long ComputeCpu2(int[] array)
        {
            var result = 0L;

            Parallel.ForEach(array, () => 0L, (value, state, local) => local + value, x =>
            {
                Interlocked.Add(ref result, x);
            });

            return result;
        }

        // CPU: (Sum) Using Linq!
        internal static long ComputeCpu3(int[] array)
        {
            return array.Aggregate(0L, (a, b) => a + b);
        }

        // CPU: (Sum) Using Parallel Linq
        internal static long ComputeCpu4(int[] array)
        {
            return array.AsParallel().Aggregate(0L, (a, b) => a + b);
        }
    }
}