#!/usr/bin/env bash

# This is a simple script to build and tag all tools packaged as Docker containers.
#
# Usage:
#   ./tools/build.sh

docker build -t compdec:compress -f ./tools/Dockerfile.compress ./tools/
