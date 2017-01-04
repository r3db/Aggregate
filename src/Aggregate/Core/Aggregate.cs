using System;
using System.Linq;
using System.Threading.Tasks;

namespace Aggregate
{
    internal static class Aggregate
    {
        // CPU: (Sum) Using Native Sequential Loop!
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

        // CPU: (Sum) Using Parallel.ForEach!
        internal static long ComputeCpu2(int[] array)
        {
            var result = 0L;
            var locker  = new object();

            Parallel.ForEach(array, () => 0L, (value, state, local) => local + value, x =>
            {
                lock (locker)
                {
                    result += x;
                }
            });

            return result;
        }

        // CPU: (Sum) Using Linq Aggregate!
        internal static long ComputeCpu3(int[] array)
        {
            return array.Aggregate(0L, (a, b) => a + b);
        }

        // CPU: (Sum) Using Linq Aggregate in Parallel!
        internal static long ComputeCpu4(int[] array)
        {
            return array.AsParallel().Aggregate(0L, (a, b) => a + b);
        }

        // Todo: Proper CPU-P without locks!
    }
}