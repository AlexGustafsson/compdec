#!/usr/bin/env bash

# Compression tool
#
# This is a tool to simplify interfacing with various compression algorithms.
# Due to its dependencies, it's preferably used via Docker. To build it run:
# ./tools/build.sh
#
# Usage:
# # Show versions of used tools
# ./tools/compress.sh versions
# # Show this help dialog
# ./tools/compress.sh help
# # Compress a file with all algorithms
# ./tools/compress.sh compress <output prefix> <input file>
# Example:
#   ./tools/compress.sh compress output/compressed-file input/test-file

# Print the versions of each included tool
function versions() {
  echo -ne "rar\t" && rar | head -n2 | tail -1
  echo -ne "gzip\t" && gzip --version | head -1
  echo -ne "zip\t" && zip --version | head -n2 | tail -1
  echo -ne "bzip2\t" && bzip2 --version 2>&1 | head -1
  echo -ne "7-zip\t" && 7z | head -n 2 | tail -1
  echo -ne "compress\t" && compress -V | head -1
  echo -ne "lz4\t" && lz4 --version
  echo -ne "brotli\t" && brotli --version
}

function compress() {
  output_path="$1"
  shift
  input_files="$*"

  # Create a directory to hold the files
  temp_directory="$(mktemp -d)"
  cp -rt "$temp_directory" $input_files

  # Tar the file
  tar_file="$(mktemp)"
  tar -cf "$tar_file" $input_files &> /dev/null

  rar a "$output_path/compressed.rar" "$temp_directory" > /dev/null

  gzip --force --stdout "$tar_file" > "$output_path/compressed.gzip"

  zip -r "$output_path/compressed.zip" "$temp_directory" > /dev/null

  bzip2 --force --stdout "$tar_file" > "$output_path/compressed.bzip2"

  7z a "$output_path/compressed.7z" "$temp_directory" > /dev/null

  command compress -c -r "$temp_directory" > "$output_path/compressed.compress"

  lz4 -f -z -c "$tar_file" > "$output_path/compressed.lz4"

  brotli --output="$output_path/compressed.brotli" --force "$tar_file"

  rm -r "$temp_directory"
  rm "$tar_file"
}

function help() {
  tail -n +3 "$0" | head -n 15 | sed 's/^#\s\{0,\}//'
}

if [[ "$#" -eq 0 ]]; then
  help
  exit 1
fi

while [[ "$1" != "" ]]; do
  case "$1" in
    "versions")
      shift
      versions
      exit 0
      ;;
    "compress")
      shift
      compress "$@"
      exit 0
      ;;
    "help" | "-h" | "--help")
      help
      exit 0
      ;;
    *)
      help
      exit 1
      ;;
  esac
  shift
done
