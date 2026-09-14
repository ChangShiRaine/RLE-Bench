#!/bin/bash
# Reference solution (Harbor Oracle) for task07: runs the baseline policy
# for a short episode through the public dev runner, then stages the
# policy package as the deliverable.
set -eu

mkdir -p /logs/artifacts
cp -r /solution/payload/policy /logs/artifacts/

cd /workspace
python3 dev_runner.py /logs/artifacts/policy --seed 35 --budget 20 || true
echo "task07 reference solution staged."
