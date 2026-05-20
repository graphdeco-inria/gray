#!/usr/bin/env bash
set -e

i=0

for scene in 360_v2/{bicycle,flowers,garden,treehill,stump}; do 
    i=$((i+1))
    echo Creating dense point cloud for scene $scene [${i}/13]
    python third_party/edgs.py -s data/$scene -r 4 --roma_model outdoors
done

for scene in 360_v2/{bonsai,counter,kitchen,room}; do
    i=$((i+1))
    echo Creating dense point cloud for scene $scene [${i}/13]
    python third_party/edgs.py -s data/$scene -r 2 --roma_model indoors
done

for scene in tandt/{train,truck}; do
    i=$((i+1))
    echo Creating dense point cloud for scene $scene [${i}/13]
    python third_party/edgs.py -s data/$scene -r 1 --roma_model outdoors
done

for scene in db/{drjohnson,playroom}; do
    i=$((i+1))
    echo Creating dense point cloud for scene $scene [${i}/13]
    python third_party/edgs.py -s data/$scene -r 1 --roma_model indoors
done
