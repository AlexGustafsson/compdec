#!/usr/bin/env bash

# Dataset creation tool
#
# This is a tool to simplify creating the dataset (compressing GovDocs).
#
# Usage:
#   ./tools/create-dataset.sh <base-dir> <target-dir>
# Examples:
#   ./tools/create-dataset.sh ./data/govdocs ./data/dataset
#   # Only compress part of the dataset
#   MAXIMUM_FILES=10 ./tools/create-dataset.sh ./data/govdocs ./data/dataset

MAXIMUM_FILES="${MAXIMUM_FILES:-0}"

function help() {
  tail -n +3 "$0" | head -n 8 | sed 's/^#\s\{0,\}//'
}

if [[ ! "$#" -eq 2 ]]; then
  help
  exit 1
fi

base_dir="$(realpath "$1")"
target_dir="$(realpath "$2")"

files="$(find "$base_dir" -type f)"

processed_files=0

while read -r file; do
  id="$(basename "$file")"
  id="${id%.*}"
  output_directory="$target_dir/$id"
  mkdir -p "$output_directory"
  docker run --detach --rm --volume "$file:/var/input" --volume "$output_directory:/var/output" compdec:compress compress "/var/output" "/var/input"

  processed_files=$(($processed_files + 1))
  if [[ $processed_files -eq $MAXIMUM_FILES ]]; then
    break
  fi

  # Sleep some time to not start thousands of containers. One run typically takes half a second or so
  # Sleep every 10th file
  if [[ $(($processed_files % 10)) -eq 0 ]]; then
    sleep 1
  fi
done<<<"$files"

echo "Compressed $processed_files files"
