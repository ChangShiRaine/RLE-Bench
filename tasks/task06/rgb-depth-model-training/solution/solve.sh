#!/bin/bash
# Reference solution (Harbor Oracle) for rgb-depth-model-training: trains the
# reference CNN in this container on the provided GPU, then stages model.pt
# and its training report.
set -eu

mkdir -p /logs/artifacts
cd /workspace

python3 /solution/train_solution.py /logs/artifacts
echo "rgb-depth-model-training reference solution staged."
