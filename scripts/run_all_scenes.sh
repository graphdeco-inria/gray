#!/usr/bin/env bash

set -e

: ${1:?An output path must be provided as first argument}

for scene in treehill flowers garden stump bicycle; do
    bash run.sh "$1/${scene}/" -r 4 -s "data/360_v2/${scene}" "${@:2}"
done

for scene in kitchen room bonsai counter; do
    bash run.sh "$1/${scene}/" -r 2 -s "data/360_v2/${scene}" "${@:2}"
done

for path in data/db/* data/tandt/*; do
    bash run.sh "$1/$(basename "$path")/" -r 1 -s "$path" "${@:2}"
done