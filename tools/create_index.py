import mimetypes
import os
import sys
from math import ceil

def recurse(chunk_size: int, directory: str) -> None:
    for root, directories, files in os.walk(dataset_path):
        for file in files:
            file_path = os.path.abspath(os.path.join(root, file))
            file_size = os.path.getsize(file_path)
            chunks = ceil(file_size / chunk_size)
            extension = os.path.splitext(file_path)[1].replace(".", "")
            print('"{}",{},{},{},"{}"'.format(file_path, file_size, chunk_size, chunks, extension))

def main(chunk_size: int, dataset_path: str) -> None:
    print('"file path","file size","chunk size","chunks",extension')
    recurse(chunk_size, dataset_path)

if __name__ == '__main__':
    chunk_size = int(sys.argv[1])
    dataset_path = sys.argv[2]
    main(chunk_size, dataset_path)
