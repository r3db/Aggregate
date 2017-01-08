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
            const int length = 82000014;
            var data = Enumerable.Range(1, length).Select(x => x % 5).ToArray();
            var expected = data.Sum();

            Func<int, int, int> op = (a, b) => a + b;

            Measure(() => Aggregate.ComputeCpu1(data, op), expected, "CPU: Using Sequential Loop!");
            Measure(() => Aggregate.ComputeCpu2(data, op), expected, "CPU: Using Linq!");
            Measure(() => Aggregate.ComputeCpu3(data, op), expected, "CPU: Using Parallel ForEach!");
            Measure(() => Aggregate.ComputeCpu4(data, op), expected, "CPU: Using Parallel Linq!");

            Console.WriteLine(new string('\n', 5));

            Measure(() => Aggregate.ComputeGpu1(data, op), expected, "GPU: Using Alea Parallel Linq!");
            Measure(() => Aggregate.ComputeGpu2(data, op), expected, "GPU: Interleaved Addressing!");

            Console.WriteLine("Done!");
            Console.ReadLine();
        }

        private static void Measure(Func<long> func, int expected, string description)
        {
            Func<Stopwatch, string> formatElapsedTime = watch => watch.Elapsed.TotalSeconds >= 1
                ? string.Format(CultureInfo.InvariantCulture, "{0,9}s",  watch.Elapsed.TotalSeconds)
                : string.Format(CultureInfo.InvariantCulture, "{0,9}ms", watch.Elapsed.TotalMilliseconds);

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