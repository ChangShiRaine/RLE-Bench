#!/bin/bash
# Reference solution (Harbor Oracle) for rgb-depth.
set -eu

mkdir -p /logs/artifacts
cp -r /solution/payload/. /logs/artifacts/
echo "rgb-depth reference solution staged."
