using Alea;
using Alea.Parallel;
using System;
using System.Diagnostics;
using System.Globalization;
using System.Linq;

namespace Aggregate
{
    internal static class Program
    {
        private static void Main()
        {
            //const int length = 260000023;
            const int length = 82000015;
            var data = Enumerable.Range(1, length).Select(x => x % 5).ToArray();
            var expected = data.Sum();

            Func<int, int, int> op = (a, b) => a + b;

            Measure(() => AggregateCpu.Compute1(data, op), expected, "CPU: Using Sequential Loop!");
            Measure(() => AggregateCpu.Compute2(data, op), expected, "CPU: Using Linq!");
            Measure(() => AggregateCpu.Compute3(data, op), expected, "CPU: Using Parallel ForEach!");
            Measure(() => AggregateCpu.Compute4(data, op), expected, "CPU: Using Parallel Linq!");

            Measure(() => Gpu.Default.Aggregate(data, op), expected, "GPU: Using Alea Parallel Linq!");

            Measure(() => AggregateGpuIA.ComputeGpu1(data, op), expected, "GPU: Interleaved Addressing! (Recursive)");
            Measure(() => AggregateGpuIA.ComputeGpu2(data, op), expected, "GPU: Interleaved Addressing! (Loop)");

            Measure(() => AggregateGpuSA.ComputeGpu1(data, op), expected, "GPU: Sequential Addressing! (Recursive)");
            Measure(() => AggregateGpuSA.ComputeGpu2(data, op), expected, "GPU: Sequential Addressing! (Loop)");

            Console.WriteLine("Done!");
            Console.ReadLine();
        }

        private static void Measure(Func<int> func, int expected, string description)
        {
            const string format = "{0,9}";

            Func<Stopwatch, string> formatElapsedTime = watch => watch.Elapsed.TotalSeconds >= 1
                ? string.Format(CultureInfo.InvariantCulture, format +"  (s)",  watch.Elapsed.TotalSeconds)
                : watch.Elapsed.TotalMilliseconds >= 1
                    ? string.Format(CultureInfo.InvariantCulture, format + " (ms)", watch.Elapsed.TotalMilliseconds)
                    : string.Format(CultureInfo.InvariantCulture, format + " (μs)", watch.Elapsed.TotalMilliseconds * 1000000);

            Func<Stopwatch, string> bandwidth = watch => string.Format(CultureInfo.InvariantCulture, "{0:F4} GB/s", (82000015 * 4) / (watch.Elapsed.TotalMilliseconds * 1000000));

            var sw1 = Stopwatch.StartNew();
            var result1 = func();
            sw1.Stop();

            Console.WriteLine(new string('-', 43));
            Console.WriteLine(description);
            Console.ForegroundColor = result1 != expected ? ConsoleColor.Red : ConsoleColor.Cyan;
            Console.WriteLine("{0} - {1} : {2} [Cold]", result1, formatElapsedTime(sw1), bandwidth(sw1));
            Console.ResetColor();

            var sw2 = Stopwatch.StartNew();
            var result2 = func();
            sw2.Stop();
            Console.ForegroundColor = result2 != expected ? ConsoleColor.Red : ConsoleColor.Cyan;
            Console.WriteLine("{0} - {1} : {2} [Warm]", result2, formatElapsedTime(sw2), bandwidth(sw2));
            Console.ResetColor();
        }
    }
}