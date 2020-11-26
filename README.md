CompDec
======

A project in machine learning and digital forensics for the courses DV2578 (Machine Learning) and DV2579 (Advanced Course in Digital Forensics).

In digital forensics *data carving* is the act of extracting files directly from some memory media - without any metadata or known filesystem. Conventional techniques use simple heuristics such as magic numbers, headers etc. These techniques do not scale well due to a limited number of supported file types, slow processing speeds and insufficient accuracy.

Recently, machine learning has been applied to the subject, achieving state-of-the-art results both in terms of scale, accuracy and speed. These techniques utilize an efficient feature extraction from files that can be turned into a small image or other representation of the features. The images are then fed to convolutional neural networks to learn to identify parts of files.

These techniques focus on generality to identify files such as documents (.txt, .docx, .ppt, .pdf) and images (.jpg, .png). There is a gap in research when it comes to effectively identify compressed files and what algorithm was used. Compression algorithms seek to make data as dense as possible, which will in turn likely yield a higher entropy than a typical file. This in theory could make detection much harder.

This project aims to fill this gap, answering the following questions:

* How do compressed files compare to non-compressed files in terms of entropy?
* How can a machine-learning system be designed and trained to detect compression algorithms?

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
"file path","file size","chunk size", mime
"/path/to/data/dataset/thread0.zip",322469174,78728,"application/zip"
"/path/to/data/dataset/909/909820.pdf",291569,72,"application/pdf"
"/path/to/data/dataset/135/135778.pdf",14013,4,"application/pdf"
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
python3 ./tools/stratified_sampling.py <index path> <strata size>
```

Example:
```sh
python3 tools/stratified_sampling.py index.txt 20
```

Example output:
```
mime,samples,frequency
"application/pdf",257,0.26
"text/html",227,0.23
"text/plain",84,0.085
"image/jpeg",104,0.1
"application/msword",67,0.068
"text/xml",31,0.031
"application/vnd.ms-powerpoint",56,0.057
"image/gif",27,0.027
"text/csv",17,0.017
"None",29,0.029
"application/vnd.ms-excel",60,0.061
"application/postscript",22,0.022
"image/png",3,0.003
"application/x-shockwave-flash",7,0.0071
Total samples: 991
Strata size: 20
"file path","file size","chunk size", mime
"/path/to/data/dataset/672/672102.jpg",319707,79,"image/jpeg"
"/path/to/data/dataset/527/527337.pdf",121728,30,"application/pdf"
"/path/to/data/dataset/740/740380.doc",632320,155,"application/msword"
"/path/to/data/dataset/999/999354.pdf",369663,91,"application/pdf"
...
```
