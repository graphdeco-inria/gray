#!/usr/bin/env bash
set -e

mkdir -p data

echo Downloading datasets to \`data/\`...

wget -q --show-progress -P data/ http://storage.googleapis.com/gresearch/refraw360/360_v2.zip
python -m zipfile -e data/360_v2.zip data/360_v2/
rm data/360_v2.zip
rm data/360_v2/flowers.txt
rm data/360_v2/treehill.txt

wget  -q --show-progress -P data/ https://storage.googleapis.com/gresearch/refraw360/360_extra_scenes.zip
python -m zipfile -e data/360_extra_scenes.zip data/360_v2/
rm data/360_extra_scenes.zip

wget  -q --show-progress -P data/ https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip 
python -m zipfile -e data/tandt_db.zip data/
rm data/tandt_db.zip

echo All datasets downloaded.




