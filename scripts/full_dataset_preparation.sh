#!/usr/bin/env bash
set -e

bash scripts/download_all_scenes.sh
bash scripts/resize_all_scenes.sh
bash scripts/create_all_dense_point_clouds.sh