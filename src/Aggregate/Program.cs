using Alea;
using Alea.CSharp;
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
            //const int length = 4000000;
            var data = Enumerable.Range(1, length).Select(x => x % 5).ToArray();
            var expected = data.Sum();

            Func<int, int, int> op = (a, b) => a + b;

            Measure(() => AggregateCpu.Compute1(data, op), expected, false, length, "CPU: Using Sequential Loop!");
            Measure(() => AggregateCpu.Compute2(data, op), expected, false, length, "CPU: Using Linq!");
            Measure(() => AggregateCpu.Compute3(data, op), expected, false, length, "CPU: Using Parallel ForEach!");
            Measure(() => AggregateCpu.Compute4(data, op), expected, false, length, "CPU: Using Parallel Linq!");

            Measure(() => Gpu.Default.Aggregate(data, op), expected, true, length, "GPU: Using Alea Parallel Linq!");
            Measure(() => AggregateGpuIA.ComputeGpu1(data, op), expected, true, length, "GPU: Interleaved Addressing! (Loop)");
            Measure(() => AggregateGpuSA.ComputeGpu1(data, op), expected, true, length, "GPU: Sequential Addressing!");
            Measure(() => AggregateGpuSAFB.ComputeGpu1(data, op), expected, true, length, "GPU: Sequential Addressing Fully Busy!");
            Measure(() => AggregateGpuSAFBU.ComputeGpu1(data, op), expected, true, length, "GPU: Sequential Addressing Fully Busy Unroll!");
            
            Console.WriteLine("Done!");
            Console.ReadLine();
        }

        private static void Measure(Func<int> func, int expected, bool isGpu, int length, string description)
        {
            const string format = "{0,9}";

            Func<Stopwatch, string> formatElapsedTime = w => w.Elapsed.TotalSeconds >= 1
                ? string.Format(CultureInfo.InvariantCulture, format +"  (s)",  w.Elapsed.TotalSeconds)
                : w.Elapsed.TotalMilliseconds >= 1
                    ? string.Format(CultureInfo.InvariantCulture, format + " (ms)", w.Elapsed.TotalMilliseconds)
                    : string.Format(CultureInfo.InvariantCulture, format + " (μs)", w.Elapsed.TotalMilliseconds * 1000);

            Func<bool, ConsoleColor> color = error => error
                ? ConsoleColor.Red 
                : isGpu 
                    ? ConsoleColor.Green 
                    : ConsoleColor.Cyan;

            Func<Stopwatch, string> bandwidth = w => string.Format(CultureInfo.InvariantCulture, "{0,7:F4} GB/s", (length * sizeof(int)) / (w.Elapsed.TotalMilliseconds * 1000000));
            
            var sw1 = Stopwatch.StartNew();
            var result1 = func();
            sw1.Stop();

            Console.WriteLine(new string('-', 43));
            Console.WriteLine(description);
            Console.ForegroundColor = color(result1 != expected);
            Console.WriteLine("{0} - {1} - {2} [Cold]", result1, formatElapsedTime(sw1), bandwidth(sw1));
            Console.ResetColor();

            var sw2 = Stopwatch.StartNew();
            var result2 = func();
            sw2.Stop();
            Console.ForegroundColor = color(result2 != expected);
            Console.WriteLine("{0} - {1} - {2} [Warm]", result2, formatElapsedTime(sw2), bandwidth(sw2));
            Console.ResetColor();
        }
    }
}