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
    internal static class AggregateCpu
    {
        // CPU: Using Sequential Loop!
        internal static T Compute1<T>(T[] array, Func<T, T, T> op)
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
        internal static T Compute2<T>(T[] array, Func<T, T, T> op)
        {
            return array.Aggregate(op);
        }

        // CPU: Using Parallel ForEach!
        // ReSharper disable once ParameterTypeCanBeEnumerable.Global
        internal static T Compute3<T>(T[] array, Func<T, T, T> op)
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

        // CPU: Using Parallel Linq!
        // ReSharper disable once ParameterTypeCanBeEnumerable.Global
        internal static T Compute4<T>(T[] array, Func<T, T, T> op)
        {
            return array.AsParallel().Aggregate(op);
        }
    }
}