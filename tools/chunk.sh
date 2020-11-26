#!/usr/bin/env bash

# Chunking tool
#
# This is a tool to chunk a file.
#
# Usage:
#   ./tools/chunk.sh <chunk size> <input file> <output directory>
# Examples:
#   # Extract 4096B chunks from this file to the output directory
#   ./tools/chunk.sh 4096 ./tools/chunk.sh ./output

chunk_size="$1"
input_file="$2"
target_directory="$3"

if [[ ! "$#" = 3 ]]; then
  echo "Usage:"
  exit 1
fi

if [[ ! -f "$input_file" ]]; then
  echo "The input file is not accessible"
  exit 1
fi

mkdir -p "$target_directory"

file_size="$(du -b "$input_file" | cut -f1)"
blocks="$((file_size / chunk_size + 1))"

echo "file,\"chunk size\",size"
for i in $(seq 1,"$blocks"); do
  chunk_start="$(((i - 1) * $chunk_size))"
  chunk_end="$((chunk_size * i))"
  actual_chunk_size="$(($chunk_end > $file_size ? $file_size - $chunk_start : $chunk_end - $chunk_start))"
  tail -c +"$chunk_end" "$input_file" | head -c "$chunk_size" > "$target_directory/$i.chunk"
  echo "\"$input_file\",$chunk_size,$actual_chunk_size"
done
