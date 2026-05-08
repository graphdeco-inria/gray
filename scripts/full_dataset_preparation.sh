#!/usr/bin/env bash
set -e

bash scripts/download_all_datasets.sh
bash scripts/resize_all_datasets.sh
bash scripts/create_every_dense_point_cloud.sh