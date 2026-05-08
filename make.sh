#!/bin/bash

set -e

cmake -S . -B build -G Ninja "$@"
cmake --build build -- -j$(nproc)