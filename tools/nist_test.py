import os
import sys
import csv
import builtins as __builtin__

from typing import List


sys.path.insert(0, os.path.abspath("./sp800_22_tests"))

def noop(*args, **kwargs):
    pass

def bits_from_bytes(sample: bytes) -> List[int]:
    bits = []
    for byte in sample:
        for i in range(8):
            bit = (byte >> i) & 1
            bits.append(bit)
    return bits

def run_tests(sample: bytes) -> None:
    tests = [
        "monobit_test",
        "frequency_within_block_test",
        "runs_test",
        "longest_run_ones_in_a_block_test",
        "binary_matrix_rank_test",
        "dft_test",
        "non_overlapping_template_matching_test",
        "overlapping_template_matching_test",
        "maurers_universal_test",
        "linear_complexity_test",
        "serial_test",
        "approximate_entropy_test",
        "cumulative_sums_test",
        "random_excursion_test",
        "random_excursion_variant_test"
    ]

    bits = bits_from_bytes(sample)
    results = list()

    for test in tests:
        module = __import__ ("sp800_22_" + test)
        function = getattr(module, test)

        # Don't let the tests print anything
        _print = print
        __builtin__.print = noop
        (success, p, plist) = function(bits)
        __builtin__.print = _print

        summary_name = test
        if success:
            print("+", end="", flush=True)
            summary_result = "pass"
        else:
            print("-", end="", flush=True)
            summary_result = "fail"

        if p != None:
            # print("  P={}".format(p))
            summary_p = str(p)

        if plist != None:
            summary_p = str(min(plist))

        results.append((summary_name,summary_p, summary_result))

    print()

    for result in results:
        (summary_name, summary_p, summary_result) = result
        print(summary_name.ljust(39), summary_p.ljust(23), summary_result)


def main(strata_path: str) -> None:
    with open(strata_path, "r") as strata_file:
        reader = csv.reader(strata_file, delimiter=",", quotechar='"')
        # Skip header
        next(reader, None)
        for file_path, offset, chunk_size, mime in reader:
            with open(file_path, "rb") as sample_file:
                sample_file.seek(int(offset), 0)
                sample = sample_file.read(int(chunk_size))
                print(file_path)
                run_tests(sample)


if __name__ == "__main__":
    strata_path = sys.argv[1]
    main(strata_path)
