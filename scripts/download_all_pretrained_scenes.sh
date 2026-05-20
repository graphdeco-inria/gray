#!/usr/bin/env bash
set -e

mkdir -p output/pretrained

echo Downloading datasets to \`output/pretrained\`...

for scene in bicycle bonsai counter drjohnson flowers garden kitchen playroom room stump train treehill truck; do 
    wget https://repo-sam.inria.fr/nerphys/gray/pretrained/$scene.zip -O output/pretrained/$scene.zip
    python -m zipfile -e output/pretrained/$scene.zip output/pretrained/$scene
    rm output/pretrained/$scene.zip
done
