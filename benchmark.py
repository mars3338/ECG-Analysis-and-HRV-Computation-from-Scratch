"""Runs the full pipeline over the database and reports sensitivity and positive predictive
value, both per record and across them.

    python benchmark.py                 all 48 records, 30 minutes each
    python benchmark.py --aami          EC57 44-record subset, excludes paced beats
    python benchmark.py --duration 300  first 5 minutes only, much faster
"""
import sys
import time
import project

# ANSI/AAMI EC57 excludes recordings containing paced beats, because pacemaker spikes are 
# narrow high-amplitude artefacts that mislead amplitude-based detectors. Published results 
# are usually quoted on the remaining 44.
paced = ["102", "104", "107", "217"]

def main():
    exclude_paced = "--aami" in sys.argv
    if "--duration" in sys.argv:
        duration = int(sys.argv[sys.argv.index("--duration") + 1])
    else:
        duration = 1800
    started = time.time()
    results, TP, FP, FN = benchmark(duration, exclude_paced)
    print(f"{'record':>7} {'Se':>8} {'PPV':>8} {'TP':>7} {'FP':>5} {'FN':>5}")
    for result in results:
        print(f"{result[0]:>7} {result[1] * 100:7.2f}% {result[2] * 100:7.2f}% {result[3]:7d} {result[4]:5d} {result[5]:5d}", flush=True,)
    print(f"Total sensitivity = {100*TP/(TP+FN):.2f}%, Total PPV = {100*TP/(TP+FP):.2f}%")
    print(f"Completed in {time.time() - started:.0f}s")

def benchmark(duration, exclude_paced):
    """Benchmark the detector on the database or subset thereof.

    Args:
        duration (int): number of seconds to process from each record
        exclude_paced (bool): whether to exclude paced records

    Returns:
        tuple: (results, TP, FP, FN)
            results (list): list of tuples (record_name, sensitivity, ppv, tp, fp, fn)
            TP (int): total true positives across all records
            FP (int): total false positives across all records
            FN (int): total false negatives across all records
    """
    records = [r for r in project.files if not (exclude_paced and r in paced)]
    results = []
    TP = FP = FN = 0
    for record_name in records:
        record, fs, reference = project.load(record_name, 0, duration)
        peaks = project.detect(project.clean(record, fs), fs)
        sensitivity, ppv, tp, fp, fn = project.validate(peaks, reference, fs)
        results.append((record_name, sensitivity, ppv, tp, fp, fn))
        TP += tp
        FP += fp
        FN += fn
    return results, TP, FP, FN

if __name__ == "__main__":
    main()

