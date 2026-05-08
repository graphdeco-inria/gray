#!/usr/bin/env bash

set -e

for scene in treehill flowers garden stump bicycle; do
    bash run.sh "$1/${scene}/" -i images_4 -s "data/360_v2/${scene}" --eval "${@:2}"
done

for scene in kitchen room bonsai counter; do
    bash run.sh "$1/${scene}/" -i images_2 -s "data/360_v2/${scene}" --eval "${@:2}"
done

for path in data/db/* data/tandt/*; do
    bash run.sh "$1/$(basename "$path")/" -i images_1 -s "$path" --eval "${@:2}"
done