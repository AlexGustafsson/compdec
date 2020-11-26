#!/usr/bin/env bash

# GovDocs tool
#
# This is a tool to simplify communication with GovDocs:
# https://digitalcorpora.org/corpora/files.
#
# Usage:
#   ./tools/govdocs.sh download <target-directory> <file 1> [file 2] [file 3] ...
# Examples:
#   # Download a single thread (about 300MB)
#   ./tools/govdocs.sh download data threads/thread0.zip


# http://downloads.digitalcorpora.org/corpora/files/govdocs1/threads/

BASE_URL="http://downloads.digitalcorpora.org/corpora/files/govdocs1"

function clean() {
  wait < <(jobs -p)
}
trap clean SIGINT

function download() {
  source="$1"
  destination="$2"
  echo "[Download started] $BASE_URL/$source -> $destination"
  wget --quiet --output-document="$destination" "$BASE_URL/$source"
  echo "[Download complete] $BASE_URL/$source -> $destination"
}

function download_all() {
  destination="$1"
  shift
  sources="$*"
  for source in $sources; do
    mkdir -p "$destination/$(dirname $source)"
    download "$source" "$destination/$source" &
  done
}

function help() {
  tail -n +3 "$0" | head -n 10 | sed 's/^#\s\{0,\}//'
}

if [[ "$#" -eq 0 ]]; then
  help
  exit 1
fi

while [[ "$1" != "" ]]; do
  case "$1" in
    "download")
      shift
      download_all "$@"
      clean
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
