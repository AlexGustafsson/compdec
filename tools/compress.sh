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
  output_prefix="$1"
  shift
  input_files="$*"

  # Create a directory to hold the files
  temp_directory="$(mktemp -d)"
  cp -rt "$temp_directory" $input_files

  # Tar the file
  tar_file="$(mktemp)"
  tar -cf "$tar_file" $input_files &> /dev/null

  rar a "$output_prefix.rar" "$temp_directory" > /dev/null

  gzip --force --fast --stdout "$tar_file" > "$output_prefix.gzip.fast"
  gzip --force --best --stdout "$tar_file" > "$output_prefix.gzip.best"

  zip -1 -r "$output_prefix.zip.fast" "$temp_directory" > /dev/null
  zip -9 -r "$output_prefix.zip.best" "$temp_directory" > /dev/null

  bzip2 --force --fast --stdout "$tar_file" > "$output_prefix.bzip2.fast"
  bzip2 --force --best --stdout "$tar_file" > "$output_prefix.bzip2.best"

  7z a "$output_prefix.7z" "$temp_directory" > /dev/null

  command compress -c -r "$temp_directory" > "$output_prefix.compress"

  lz4 -f -z -1 -c "$tar_file" > "$output_prefix.lz4.fast"
  lz4 -f -z -9 -c "$tar_file" > "$output_prefix.lz4.best"

  brotli -0 --output="$output_prefix.brotli.fast" --force "$tar_file"
  brotli -9 --output="$output_prefix.brotli.best" --force "$tar_file"

  rm -r "$temp_directory"
  rm "$tar_file"
  ls "$(dirname "$output_prefix")"
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
