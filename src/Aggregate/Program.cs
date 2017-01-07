using System;
using System.Diagnostics;
using System.Linq;

namespace Aggregate
{
    internal static class Program
    {
        private static void Main()
        {
            //const int length = 260000023;
            const int length = 8000014;
            var data = Enumerable.Range(1, length).Select(x => x % 10).ToArray();
            var expected = data.Sum();

            Measure(() => Aggregate.ComputeCpu1(data), expected, "CPU: Using Sequential Loop!");
            Measure(() => Aggregate.ComputeCpu2(data), expected, "CPU: Using Parallel ForEach!");
            Measure(() => Aggregate.ComputeCpu3(data), expected, "CPU: Using Linq!");
            Measure(() => Aggregate.ComputeCpu4(data), expected, "CPU: Using Parallel Linq!");

            Console.WriteLine();

            Measure(() => Aggregate.ComputeGpu1(data), expected, "GPU: Using Alea Parallel Linq!");
            Measure(() => Aggregate.ComputeGpu2(data), expected, "GPU: Interleaved Addressing!");

            Console.WriteLine("Done!");
            Console.ReadLine();
        }

        private static void Measure(Func<long> func, int expected, string description)
        {
            Func<Stopwatch, string> formatElapsedTime = (watch) => watch.Elapsed.TotalSeconds >= 1
                ? $"{watch.Elapsed.TotalSeconds}s"
                : $"{watch.ElapsedMilliseconds}ms";

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
            Console.WriteLine("{0} - {1} [Cold]", result2, formatElapsedTime(sw2));
            Console.ResetColor();
        }
    }
}