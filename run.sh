#!/bin/bash

set -e

python train.py -m "${@}"
python render.py -m "$1"
python metrics.py -m "$1"
python measure_fps.py -m "$1"