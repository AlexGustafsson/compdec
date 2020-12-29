import sys
import csv
import random

from typing import Dict, List, Tuple

# path, offset, chunk size, mime
Sample = Tuple[str, int, int, str]

# mime: Sample
samples: Dict[str, List[Sample]] = {}

def main(seed: str, index_path: str, strata_size: int) -> None:
    random.seed(seed)

    with open(index_path) as file:
        reader = csv.reader(file, delimiter=",", quotechar='"')
        # Skip header
        next(reader, None)
        skipped_samples = 0
        for file_path, file_size, chunk_size, chunks, extension in reader:
            file_size = int(file_size)
            chunk_size = int(chunk_size)
            chunks = int(chunks)
            if extension not in samples:
                samples[extension] = []
            for i in range(0, chunks):
                offset = i * chunk_size
                if offset + chunk_size >= file_size:
                    skipped_samples += 1
                    break
                samples[extension].append((file_path, offset, chunk_size, extension))
        print("Warning: skipping {} uneven chunks".format(skipped_samples), file=sys.stderr)

    # Total number of chunks for all samples
    total_samples = sum([len(samples[mime]) for mime in samples])
    strata: List[Sample] = []
    print("extension,samples,frequency", file=sys.stderr)
    for mime in samples:
        size = len(samples[mime])
        frequency = size / total_samples
        print("\"{0}\",{1},{2:.2}".format(mime, size, frequency), file=sys.stderr)
        # Randomize each file type
        random.shuffle(samples[mime])
        # Pick the correct number of samples
        strata += samples[mime][:round(frequency * strata_size)]
    print("Total samples:", total_samples, file=sys.stderr)
    print("Strata size:", len(strata), file=sys.stderr)

    # Randomize the sample
    random.shuffle(strata)

    print('"file path",offset,"chunk size",extension')
    for sample in strata:
        print('"{}",{},{},"{}"'.format(*sample))


if __name__ == '__main__':
    seed = sys.argv[1]
    index_path = sys.argv[2]
    strata_size = int(sys.argv[3])
    main(seed, index_path, strata_size)
