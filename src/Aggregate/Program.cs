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
            const int length = 8000013;
            var data = Enumerable.Range(1, length).Select(x => x % 10).ToArray();

            Measure(() => Aggregate.ComputeCpu1(data), "CPU: Using Sequential Loop!");
            Measure(() => Aggregate.ComputeCpu2(data), "CPU: Using Parallel ForEach!");
            Measure(() => Aggregate.ComputeCpu3(data), "CPU: Using Linq!");
            //Measure(() => Aggregate.ComputeCpu4(data), "CPU: Using Parallel Linq!");
            Measure(() => Aggregate.ComputeGpu1(data), "GPU: Using Parallel Linq!");
            Measure(() => Aggregate.ComputeGpu2(data), "GPU: XXX");

            Console.WriteLine("Expected: {0:D}", length * ((long)length + 1) / 2);
            // In case of overflow we display display. Alea does not have Aggregates has flexible as traditional Linq.
            Console.WriteLine("Expected: {0:D}", unchecked((int)(length * ((long)length + 1) / 2)));

            Console.WriteLine("Done!");
            Console.ReadLine();
        }

        private static void Measure(Func<long> func, string description)
        {
            Func<Stopwatch, string> formatElapsedTime = (watch) => watch.Elapsed.TotalSeconds >= 1
                ? $"{watch.Elapsed.TotalSeconds}s"
                : $"{watch.ElapsedMilliseconds}ms";

            var sw1 = Stopwatch.StartNew();
            var result1 = func();
            sw1.Stop();

            Console.WriteLine(new string('-', 43));
            Console.WriteLine(description);
            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.WriteLine("{0} - {1} [Cold]", result1, formatElapsedTime(sw1));
            Console.ResetColor();

            var sw2 = Stopwatch.StartNew();
            var result2 = func();
            sw2.Stop();
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine("{0} - {1} [Cold]", result2, formatElapsedTime(sw2));
            Console.ResetColor();

            //Console.WriteLine(new string('-', 43));
            //Console.WriteLine(new string('\n', 4));
        }
    }
}