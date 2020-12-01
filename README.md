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

[Development](#development)<br />
[Development - Quickstart](#development-quickstart)<br />
[Development - Tools](#development-tools)

## Development
<a name="development"></a>

### Quickstart
<a name="development-quickstart"></a>

To start, first clone this repository.

```sh
git clone https://github.com/AlexGustafsson/compdec.git && cd compdec
```

To train the model, you'll need some training data. The paper uses the [GovDocs](https://digitalcorpora.org/corpora/files) dataset, but any larger dataset with a wide variety of files should work fine. For ease of use, a tool is included to download the data. The commands below download a small subset of the dataset, suitable for testing and developing.

```sh
mkdir -p data
./tools/govdocs.sh download data threads/thread0.zip
unzip -d data/govdocs data/thread0.zip
```

Now we'll need a index of the dataset, what files there are and how large they are. This is easily created using the following command. In this case we're picking chunks of maximum 4096 bytes, a common chunk size of current file systems.

```sh
python3 ./tools/create_index.py 4096 ./data/govdocs > index.csv
```

With the index created, one can perform stratified sampling to extract a sample from the population with the following command. In this case we're picking a strata of 20 samples.

```sh
python3 ./tools/stratified_sampling.py index.csv 20 > strata.csv
```

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
"file path","file size","chunk size","chunks",mime
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
mime,samples,frequency
"application/zip",78728,0.35
"application/pdf",37438,0.17
"text/html",3590,0.016
"text/plain",45112,0.2
"image/jpeg",9875,0.044
"application/msword",6659,0.03
"text/xml",598,0.0027
"application/vnd.ms-powerpoint",29038,0.13
"image/gif",580,0.0026
"text/csv",679,0.003
"None",1275,0.0057
"application/vnd.ms-excel",6953,0.031
"application/postscript",2535,0.011
"image/png",604,0.0027
"application/x-shockwave-flash",362,0.0016
Total samples: 224026
Strata size: 20
"file path",offset,"chunk size",mime
"/path/to/compdec/data/dataset/thread0.zip",108646400,4096,"application/zip"
"/path/to/compdec/data/dataset/191/191969.txt",125845504,4096,"text/plain"
"/path/to/compdec/data/dataset/354/354930.doc",307200,4096,"application/msword"
"/path/to/compdec/data/dataset/thread0.zip",34136064,4096,"application/zip"
...
```

#### compress.sh

This is a tool to simplify interfacing with various compression algorithms. Due to its dependencies, it's preferably used via Docker. To build it run: `./tools/build.sh`.

Instead of `./tools/compress.sh`, you may use `docker run -it compdec:compress`.

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
