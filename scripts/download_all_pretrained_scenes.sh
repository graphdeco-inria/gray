#!/usr/bin/env bash
set -e

mkdir -p output/pretrained

echo Downloading scenes to \`output/pretrained\`...

for scene in bicycle bonsai counter drjohnson flowers garden kitchen playroom room stump train treehill truck; do 
    curl -fL --progress-bar -o output/pretrained/$scene.zip https://repo-sam.inria.fr/nerphys/gray/pretrained/$scene.zip
    python -m zipfile -e output/pretrained/$scene.zip output/pretrained/
    rm output/pretrained/$scene.zip
done
