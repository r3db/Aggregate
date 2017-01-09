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

            Measure(() => Aggregate.ComputeCpu1(data, op), expected, "CPU: Using Sequential Loop!");
            Measure(() => Aggregate.ComputeCpu2(data, op), expected, "CPU: Using Linq!");
            Measure(() => Aggregate.ComputeCpu3(data, op), expected, "CPU: Using Parallel ForEach!");
            Measure(() => Aggregate.ComputeCpu4(data, op), expected, "CPU: Using Parallel Linq!");

            Console.WriteLine(new string('\n', 5));

            Measure(() => Aggregate.ComputeGpu1(data, op), expected, "GPU: Using Alea Parallel Linq!");
            Measure(() => Aggregate.ComputeGpu2(data, op), expected, "GPU: Interleaved Addressing! (Recursive)");
            Measure(() => Aggregate.ComputeGpu3(data, op), expected, "GPU: Interleaved Addressing! (Loop)");
            Measure(() => Aggregate.ComputeGpu4(data, op), expected, "GPU: Interleaved Addressing! (Method)");

            Measure(() => Aggregate.ComputeGpu5(data, op), expected, "GPU: Sequential Addressing! (Recursive)");
            Measure(() => Aggregate.ComputeGpu6(data, op), expected, "GPU: Sequential Addressing! (Loop)");
            Measure(() => Aggregate.ComputeGpu7(data, op), expected, "GPU: Sequential Addressing! (Method)");

            Console.WriteLine("Done!");
            Console.ReadLine();
        }

        private static void Measure(Func<long> func, int expected, string description)
        {
            const string format = "{0,9}";

            Func<Stopwatch, string> formatElapsedTime = watch => watch.Elapsed.TotalSeconds >= 1
                ? string.Format(CultureInfo.InvariantCulture, format +"  (s)",  watch.Elapsed.TotalSeconds)
                : watch.Elapsed.TotalMilliseconds >= 1
                    ? string.Format(CultureInfo.InvariantCulture, format + " (ms)", watch.Elapsed.TotalMilliseconds)
                    : string.Format(CultureInfo.InvariantCulture, format + " (μs)", watch.Elapsed.TotalMilliseconds * 1000000);

            var sw1 = Stopwatch.StartNew();
            var result1 = func();
            sw1.Stop();

            Console.WriteLine(new string('-', 43));
            Console.WriteLine(description);
            Console.ForegroundColor = result1 != expected ? ConsoleColor.Red : ConsoleColor.Cyan;
            Console.WriteLine("{0} - {1} [Cold]", result1, formatElapsedTime(sw1));
            Console.ResetColor();

            var sw2 = Stopwatch.StartNew();
            var result2 = func();
            sw2.Stop();
            Console.ForegroundColor = result2 != expected ? ConsoleColor.Red : ConsoleColor.Cyan;
            Console.WriteLine("{0} - {1} [Warm]", result2, formatElapsedTime(sw2));
            Console.ResetColor();
        }
    }
}