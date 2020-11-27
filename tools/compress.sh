#!/usr/bin/env bash

# Compression tool
#
# This is a tool to simplify interfacing with various compression algorithms
# available in the compdec:compress container (Dockerfile.compress).
#
# Usage:
#   TODO
# Example:
#   TODO

# Check if the image exists
if [[ ! "$(docker image ls compdec:compress | grep -c '^compdec')" -eq 1 ]]; then
  echo "Unable to find the compdec:compress image"
  echo "Build it using ./tools/build.sh"
  exit 1
fi

function run() {
  docker run --rm -it compdec:compress bash -c "$@"
}

# Print the versions of each included tool
function versions() {
  commands="$(cat <<-EOF
  echo -ne "rar\t" && rar | head -n2 | tail -1;
  echo -ne "gzip\t" && gzip --version | head -1;
  echo -ne "zip\t" && zip --version | head -n2 | tail -1;
  echo -ne "bzip2\t" && bzip2 --version 2>&1 | head -1;
  echo -ne "7-zip\t" && 7z | head -n 2 | tail -1;
  echo -ne "compress\t" && compress -V | head -1;
  echo -ne "lz4\t" && lz4 --version;
  echo -ne "brotli\t" && brotli --version
EOF
)"
  run "$commands"
}
