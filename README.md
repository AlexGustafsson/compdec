CompDec
======

A project in machine learning and digital forensics for the courses DV2578 (Machine Learning) and DV2579 (Advanced Course in Digital Forensics).

In digital forensics *data carving* is the act of extracting files directly from some memory media - without any metadata or known filesystem. Conventional techniques use simple heuristics such as magic numbers, headers etc. These techniques do not scale well due to a limited number of supported file types, slow processing speeds and insufficient accuracy.

Recently, machine learning has been applied to the subject, achieving state-of-the-art results both in terms of scale, accuracy and speed. These techniques utilize an efficient feature extraction from files that can be turned into a small image or other representation of the features. The images are then fed to convolutional neural networks to learn to identify parts of files.

These techniques focus on generality to identify files such as documents (.txt, .docx, .ppt, .pdf) and images (.jpg, .png). There is a gap in research when it comes to effectively identify compressed files and what algorithm was used. Compression algorithms seek to make data as dense as possible, which will in turn likely yield a higher entropy than a typical file. This in theory could make detection much harder.

This project aims to fill this gap, answering the following questions:

* How do compressed files compare to non-compressed files in terms of entropy?
* How can a machine-learning system be designed and trained to detect compression algorithms?

**TL;DR** CompDec is a novel approach to automatically detect the compression algorithm used for file fragments using machine learning.

## Table of Contents

[Dataset](#dataset)<br />
[Development](#development)<br />
[Development - Quickstart](#development-quickstart)<br />
[Development - Tools](#development-tools)

## Dataset
<a name="dataset"></a>

### Samples

In the samples directory are file chunks, visualizations and NIST Statistical tests performed on the dataset.

Below is an example visualization and NIST test for the 7-zip tool.

<p align="center">
  <img src="./samples/visualizations/7z.png">
</p>

```
...
SUMMARY
-------
monobit_test                             0.23712867340389365 PASS
frequency_within_block_test              0.28036273314388394 PASS
runs_test                                0.11846733945572493 PASS
longest_run_ones_in_a_block_test         0.5251306363531703 PASS
binary_matrix_rank_test                  0.0                FAIL
dft_test                                 0.753290157881333  PASS
non_overlapping_template_matching_test   0.9999999736364428 PASS
overlapping_template_matching_test       0.0                FAIL
maurers_universal_test                   0.0                FAIL
linear_complexity_test                   0.0                FAIL
serial_test                              0.1862667243373838 PASS
approximate_entropy_test                 0.18385318163162168 PASS
cumulative_sums_test                     0.17770673343194865 PASS
random_excursion_test                    0.24443855795386374 PASS
random_excursion_variant_test            0.013229883923921373 PASS
```

There are two pseudo-random samples, `random` and `urandom` taken from `/dev/random` and `/dev/urandom` respectively. There is also a true random sample, `true-random` taken from random.org. These random samples have one NIST test report each, available in the `.txt` file with the same name. Each "random" and random sample consists of 4096 bytes.

## Development
<a name="development"></a>

### Quickstart
<a name="development-quickstart"></a>

To start, first clone this repository.

```sh
git clone --recurse-submodules https://github.com/AlexGustafsson/compdec.git && cd compdec
```

To train the model, you'll need some training data. The paper uses the [GovDocs](https://digitalcorpora.org/corpora/files) dataset, but any larger dataset with a wide variety of files should work fine. For ease of use, a tool is included to download the data. The commands below download a small subset of the dataset, suitable for testing and developing. This procedure can be repeated for any number of available threads.

```sh
mkdir -p data
./tools/govdocs.sh download data threads/thread0.zip
unzip -d data/govdocs data/threads/thread0.zip
```

Given the base data, we can now compress it using the available tools. These tools require Docker and the Docker images available as part of this project. Build and tag them using `./tools/build.sh`.

```sh
./tools/create-dataset.sh ./data/govdocs ./data/dataset
```

Now we'll need an index of the dataset, what files there are and how large they are. This is easily created using the following command. In this case we're picking chunks of maximum 4096 bytes, a common chunk size of commonly used file systems.

```sh
python3 ./tools/create_index.py 4096 ./data/dataset > ./data/index.csv
```

As part of our analysis we want to study the entropy of compressed files. This can be done by first creating a stratified sample.

With the index created, one can perform stratified sampling to extract a sample from the population with the following command. In this case we're picking a strata of 20 samples and we're using the seed `seed`.

```sh
python3 ./tools/stratified_sampling.py seed ./data/index.csv 20 > ./data/strata.csv
```

Using the stratified sample, we can run the NIST statistical test suite on them using the following command:

```sh
python3 ./tools/nist_test.py ./data/strata.csv > ./data/tests.txt
```

We can now create two stratas, one for training and one for evaluation. This can be done using the same tool as previously.

```sh
python3 ./tools/stratified_sampling.py seed ./data/index.csv 80 > ./data/train-strata.csv
python3 ./tools/stratified_sampling.py seed ./data/index.csv 20 ./data/train-strata.csv > ./data/test-strata.csv
```

Make sure that you apply an appropriate split of the data. Although a small number was used in this example, you may use the full sample size of the dataset.

Given the dataset, we can now train a model like so:

```sh
python3 ./models/cnn.py ./data/train-strata.csv ./data/test-strata.csv my-model-name
```

The training will create a checkpoints file under `./data/checkpoints/my-model-name`. The trained model will be created in `./data/models/my-model-name.h5`. The model will overwrite any file by the same name that may exist.

To enable TensorBoard, set `use_tensorboard` to `True` and run the following command:

```sh
# --bind_all optional. Makes the site available to the local network
tensorboard --logdir ./data/tensorboard --bind_all
```

With the model trained we can predict the algorithm of a file or chunk using the following script:

```sh
python3 ./models/predict.py ./data/models/my-model-name.h5 ./data/dataset/sample
```

We'll get an output like so;

```
7z       : 0.34%
brotli   : 95.39%
bzip2    : 0.20%
compress : 0.06%
gzip     : 3.07%
lz4      : 0.57%
rar      : 0.27%
zip      : 0.09%
```

The prediction utility requires at least as many bytes as the model was trained with. By default this is 4096 bytes, but it can be changed.

### Tools
<a name="development-tools"></a>

#### chunk.sh

Chunking tool for splitting a file into chunks.

Usage:
```sh
./tools/chunk.sh <chunk size> <input file> <output directory>
```

Example:
```sh
# Extract 4096B chunks from this file to the output directory
./tools/chunk.sh 4096 ./tools/chunk.sh ./output
```

Example output:
```
file,"chunk size",size
"./tools/chunk.sh",4096,999
```

#### create_index.py

Create a index for the dataset.

Usage:
```sh
python3 tools/create_index.py <chunk size> <input directory>
```

Example:
```sh
python3 tools/create_index.py 4096 ./data/dataset
```

Example output:
```
"file path","file size","chunk size","chunks",extension
"/path/to/compdec/data/dataset/thread0.zip",322469174,4096,78728,"application/zip"
"/path/to/compdec/data/dataset/909/909820.pdf",291569,4096,72,"application/pdf"
"/path/to/compdec/data/dataset/135/135778.pdf",14013,4096,4,"application/pdf"
"/path/to/compdec/data/dataset/135/135495.html",18127,4096,5,"text/html"
...
```

#### govdocs.sh

This is a tool to simplify communication with GovDocs: https://digitalcorpora.org/corpora/files.

Usage:
```sh
./tools/govdocs.sh download <target-directory> <file 1> [file 2] [file 3] ...
```

Example:
```sh
# Download a single thread (about 300MB)
./tools/govdocs.sh download data threads/thread0.zip
```

Example output:
```
[Download started] http://downloads.digitalcorpora.org/corpora/files/govdocs1/threads/thread0.zip -> data/threads/thread0.zip
[Download complete] http://downloads.digitalcorpora.org/corpora/files/govdocs1/threads/thread0.zip -> data/threads/thread0.zip
```

#### stratified_sampling.py

This is a tool to perform a stratified sampling of a dataset.

Usage:
```sh
python3 ./tools/stratified_sampling.py <seed> <index path> <strata size>
```

Example:
```sh
python3 tools/stratified_sampling.py 1.3035772690 index.txt 20
```

Example output:
```
extension,samples,frequency
"zip",78728,0.35
"pdf",37438,0.17
"html",3590,0.016
"txt",45112,0.2
"jpeg",9875,0.044
"docx",6659,0.03
"xml",598,0.0027
"ppt",29038,0.13
"gif",580,0.0026
"csv",679,0.003
"xls",6953,0.031
"ps",2535,0.011
"png",604,0.0027
"flash",362,0.0016
Total samples: 224026
Strata size: 20
"file path",offset,"chunk size",extension
"/path/to/compdec/data/dataset/thread0.zip",108646400,4096,"zip"
"/path/to/compdec/data/dataset/191/191969.txt",125845504,4096,"txt"
"/path/to/compdec/data/dataset/354/354930.doc",307200,4096,"docx"
"/path/to/compdec/data/dataset/thread0.zip",34136064,4096,"zip"
...
```

#### compress.sh

This is a tool to simplify interfacing with various compression algorithms. Due to its dependencies, it's preferably used via Docker. To build it run: `./tools/build.sh`.

Instead of `./tools/compress.sh`, you may use `docker run -it --rm compdec:compress`.

Usage:
```
# Show versions of used tools
./tools/compress.sh versions
# Show this help dialog
./tools/compress.sh help
# Compress a file with all algorithms
./tools/compress.sh compress <output prefix> <input file>
```

Example:
```
./tools/compress.sh compress output/compressed-file input/test-file
```

#### create-dataset.sh

This is a tool to simplify creating the dataset (compressing GovDocs).

Usage:
```
./tools/create-dataset.sh <base-dir> <target-dir>
```

Examples:

```
./tools/create-dataset.sh ./data/govdocs ./data/dataset
# Only compress part of the dataset
MAXIMUM_FILES=10 ./tools/create-dataset.sh ./data/govdocs ./data/dataset
```

#### nist_test.py

This is a tool to perform the NIST statistical test suite on samples.

Usage:
```
python3 ./tools/nist_test.py ./data/strata.csv
```
