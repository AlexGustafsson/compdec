import sys
import csv
from random import shuffle

from typing import Dict, List, Tuple

# path, file_size, chunks, mime
Sample = Tuple[str, int, int, str]

# mime: Sample
samples: Dict[str, List[Sample]] = {}

def main(index_path: str, strata_size: int) -> None:
    with open(index_path) as file:
        reader = csv.reader(file, delimiter=",", quotechar='"')
        # Skip header
        next(reader, None)
        for path, file_size, chunks, mime in reader:
            if mime not in samples:
                samples[mime] = []
            samples[mime].append((path, int(file_size), int(chunks), mime))

    total_samples = sum([len(samples[mime]) for mime in samples])
    strata: List[Sample] = []
    print("mime,samples,frequency", file=sys.stderr)
    for mime in samples:
        size = len(samples[mime])
        frequency = size / total_samples
        print("\"{0}\",{1},{2:.2}".format(mime, size, frequency), file=sys.stderr)
        # Randomize each file type
        shuffle(samples[mime])
        # Pick the correct number of samples
        strata += samples[mime][:round(frequency * strata_size)]
    print("Total samples:", total_samples, file=sys.stderr)
    print("Strata size:", len(strata), file=sys.stderr)

    # Randomize the sample
    shuffle(strata)

    print('"file path","file size","chunk size", mime')
    for sample in strata:
        print('"{}",{},{},"{}"'.format(*sample))


if __name__ == '__main__':
    index_path = sys.argv[1]
    strata_size = int(sys.argv[2])
    main(index_path, strata_size)