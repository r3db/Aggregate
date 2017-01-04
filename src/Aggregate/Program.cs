using System;
using System.Diagnostics;
using System.Linq;

namespace Aggregate
{
    internal static class Program
    {
        private static void Main()
        {
            const int length = 260000000;
            var data = Enumerable.Range(1, length).ToArray();

            Measure(() => Aggregate.ComputeCpu1(data), "CPU: (Sum) Using Native Sequential Loop!");
            Measure(() => Aggregate.ComputeCpu2(data), "CPU: (Sum) Using Parallel.ForEach!");
            Measure(() => Aggregate.ComputeCpu3(data), "CPU: (Sum) Using Linq Aggregate!");
            Measure(() => Aggregate.ComputeCpu4(data), "CPU: (Sum) Using Linq Aggregate in Parallel!");

            Console.WriteLine("Expected: {0:D}", length * ((long)length + 1) / 2);

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

            Console.WriteLine();

            var sw2 = Stopwatch.StartNew();
            var result2 = func();
            sw2.Stop();
            Console.WriteLine(description);
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine("{0} - {1} [Cold]", result2, formatElapsedTime(sw2));
            Console.ResetColor();

            Console.WriteLine(new string('-', 43));
            Console.WriteLine();
        }
    }
}