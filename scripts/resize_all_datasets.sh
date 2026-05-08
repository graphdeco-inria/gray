#!/usr/bin/env bash
set -e

i=0
for scene in data/{360_v2,db,tandt}/*/; do 
    i=$((i+1))
    echo Resizing scene $scene [${i}/13]
    python resize.py -s $scene --yes
done

